from keras.models import Model
from keras.layers import Input, Dense, Flatten, Convolution2D, Permute
from keras.initializers import VarianceScaling
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
import os
import numpy as np
import gym
from gym.wrappers.monitor import Monitor

from .preprocessing import AtariProcessor
from .policy import CheckpointAnnealedPolicy
from .callbacks import MyTrainLogger, ReloadModelIntervalCheckpoint


class KerasModelBuilder:

    def __init__(self, input_shape, input_window_length, action_number, hidden_layer_size, random_seed=123):
        self.input_shape = (input_window_length,) + input_shape
        self.action_number = action_number
        self.hidden_layer_size = hidden_layer_size
        self.random_seed = random_seed

    def build(self):
        # input_shape = (28, 28) -> last convolution output shape is (1, 1, hidden_layer_size)
        input_layer = Input(self.input_shape)
        """
        Model structure from: https://github.com/fg91/Deep-Q-Learning
        """
        x = Permute((2, 3, 1))(input_layer)
        x = Convolution2D(32, (8, 8), strides=4, padding="valid", activation="relu",
                          kernel_initializer=VarianceScaling(scale=2.0, seed=self.random_seed), use_bias=False)(x)
        x = Convolution2D(64, (4, 4), strides=2, padding="valid", activation="relu",
                          kernel_initializer=VarianceScaling(scale=2.0, seed=self.random_seed), use_bias=False)(x)
        x = Convolution2D(64, (3, 3), strides=1, padding="valid", activation="relu",
                          kernel_initializer=VarianceScaling(scale=2.0, seed=self.random_seed), use_bias=False)(x)
        x = Convolution2D(self.hidden_layer_size, (7, 7), strides=1, padding="valid", activation="relu",
                          kernel_initializer=VarianceScaling(scale=2.0, seed=self.random_seed), use_bias=False)(x)
        x = Flatten()(x)
        # Will be removed if used with dueling, seems strange.
        x = Dense(self.action_number, activation='linear')(x)
        return Model(input_layer, x)


def make_deep_q_network(env, args):
    model = KerasModelBuilder(input_shape=args["input_shape"],
                              input_window_length=args["input_window_length"],
                              action_number=env.action_space.n,
                              hidden_layer_size=args["hidden_layer_size"],
                              random_seed=args["random_seed"]).build()

    memory = SequentialMemory(limit=args["replay_memory_size"], window_length=args["input_window_length"])
    processor = AtariProcessor(args["input_shape"])

    policy = CheckpointAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                      value_max1=args["starting_epslon"], value_min1=args["annealed_epslon1"],
                                      value_max2=args["annealed_epslon1"], value_min2=args["annealed_epslon2"],
                                      value_test=args["annealed_epslon2"], nb_steps1=args["annealed_steps1"],
                                      nb_steps2=args["annealed_steps2"], starting_step=args["starting_step"])

    dqn = DQNAgent(model=model, nb_actions=env.action_space.n, policy=policy, memory=memory,
                   processor=processor, nb_steps_warmup=args["replay_memory_starting_size"],
                   gamma=args["discount_factor"], target_model_update=args["target_update_frequency"],
                   enable_dueling_network=args["dueling"], enable_double_dqn=args["double_dqn"],
                   train_interval=args["gradient_update_frequency"], delta_clip=1.)

    dqn.compile(Adam(lr=args["learning_rate"]), metrics=['mae'])
    return dqn


def train_and_evaluate(args, monitor_path, checkpoint_step_filename,
                       checkpoint_weights_filename, weights_filename, log_filename):

    env = gym.make(args["env_name"])
    env = Monitor(env, monitor_path, resume=True, uid=args["run_id"],
                  video_callable=lambda episode_num: episode_num % args["record_video_every"] == 0)
    np.random.seed(args["random_seed"])
    env.seed(args["random_seed"])
    starting_step = 0
    if os.path.exists(checkpoint_step_filename):
        with open(checkpoint_step_filename, 'r') as f:
            starting_step = int(f.read())
    args["starting_step"] = starting_step
    dqn = make_deep_q_network(env, args)
    if args["starting_step"] > 0:
        dqn.load_weights(checkpoint_weights_filename)

    callbacks = [ReloadModelIntervalCheckpoint(checkpoint_weights_filename,
                                               step_path=checkpoint_step_filename,
                                               interval=args["checkpoint_frequency"],
                                               starting_step=starting_step,
                                               job_dir=args["job_dir"]),
                 MyTrainLogger(args["checkpoint_frequency"], args["training_steps"], args["job_dir"], starting_step, log_filename)]

    if args["mode"] == "Train":
        dqn.fit(env, callbacks=callbacks, verbose=0,
                nb_steps=args["training_steps"] - starting_step,
                nb_max_start_steps=args["strarting_fire_steps"], start_step_policy=lambda obs: 1)  # 1 is fire action

        dqn.save_weights(weights_filename, overwrite=True)
    else:
        dqn.load_weights(weights_filename)

    env = gym.make(args["env_name"])
    env = Monitor(env, monitor_path, resume=True, uid=args["run_id"] + "_test")
    np.random.seed(args["random_seed"])
    env.seed(args["random_seed"])
    dqn.test(env, nb_episodes=1, visualize=False,
             nb_max_start_steps=args["strarting_fire_steps"], start_step_policy=lambda obs: 1)  # 1 is fire action
