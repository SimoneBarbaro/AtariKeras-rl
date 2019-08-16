import sys
import timeit
import numpy as np
from rl.callbacks import Callback, ModelIntervalCheckpoint


class MyTrainLogger(Callback):
    def __init__(self, interval, training_steps, starting_step=0, log_filename=None):
        super(MyTrainLogger, self).__init__()
        self.interval = interval
        self.step = starting_step
        self.training_steps = training_steps
        self.episode_step = 0
        self.episode_number = 0
        self.log_filename = log_filename
        self.reset()

    def reset(self):
        """ Reset statistics """
        self.interval_start = timeit.default_timer()
        self.episode_rewards = []

    def on_train_begin(self, logs):
        """ Initialize training statistics at beginning of training """
        self.train_start = timeit.default_timer()
        print('Training for {} steps ...'.format(self.params['nb_steps']))

    def on_train_end(self, logs):
        """ Print training duration at end of training """
        duration = timeit.default_timer() - self.train_start
        print('done, took {:.3f} seconds'.format(duration))

    def on_step_begin(self, step, logs):
        if self.step % self.interval == 0:
            if len(self.episode_rewards) > 0:
                print('\n{} episodes - reward statistics: {:.3f} [{:.3f}, {:.3f}]\n'.format(len(self.episode_rewards), np.mean(self.episode_rewards), np.min(self.episode_rewards), np.max(self.episode_rewards)))
                with open(self.log_filename, "a") as f:
                    f.write("step: {}, [last {} episodes reward statistics: {:.3f} [{:.3f}, {:.3f}]]\n".format(self.step, len(self.episode_rewards), np.mean(self.episode_rewards), np.min(self.episode_rewards), np.max(self.episode_rewards)))
            self.reset()

    def on_step_end(self, step, logs):
        self.step += 1
        self.episode_step += 1

    def on_episode_end(self, episode, logs):
        """ Update reward value at the end of each episode """
        self.episode_rewards.append(logs['episode_reward'])
        print("\r{}/{} steps, episode {} ({} steps), reward: {}".format(self.step, self.training_steps, self.episode_number, self.episode_step, logs['episode_reward']), end="")
        sys.stdout.flush()
        self.episode_number += 1
        self.episode_step = 0


class ReloadModelIntervalCheckpoint(ModelIntervalCheckpoint):
    def __init__(self, checkpoint_path, step_path, interval, starting_step=0, verbose=0):
        super(ReloadModelIntervalCheckpoint, self).__init__(checkpoint_path, interval, verbose)
        self.total_steps = starting_step
        self.step_path = step_path

    def on_step_end(self, step, logs={}):
        super(ReloadModelIntervalCheckpoint, self).on_step_end(step, logs)
        if self.total_steps % self.interval != 0:
            return
        with open(self.step_path, 'w') as f:
            f.write('{}'.format(self.total_steps))
