from PIL import Image
import numpy as np
from rl.core import Processor


class AtariProcessor(Processor):

    def __init__(self, input_shape):
        super(AtariProcessor, self).__init__()
        self.input_shape = input_shape

    def process_observation(self, observation):
        assert observation.ndim == 3
        img = Image.fromarray(observation)
        img = img.resize(self.input_shape).convert('L')
        processed_observation = np.array(img)
        assert processed_observation.shape == self.input_shape
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)
