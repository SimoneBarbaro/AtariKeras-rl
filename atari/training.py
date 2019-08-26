import os
import numpy as np
import gym
from gym.wrappers.monitor import Monitor
from .model import make_deep_q_network
from .callbacks import MyTrainLogger, ReloadModelIntervalCheckpoint


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
                 MyTrainLogger(interval=args["checkpoint_frequency"], training_steps=args["training_steps"],
                               starting_step=starting_step, job_dir=args["job_dir"], log_filename=log_filename,
                               monitor_dir=monitor_path, monitor_interval=args["record_video_every"])]

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
