from __future__ import print_function
__author__ = "Abhimanyu Banerjee"

import numpy as np
from random import sample
import pdb

class ExperienceReplay(object):

	def __init__(self, memory_size=100):
		self.memory = []
		self._memory_size = memory_size

	''' optional stuff ...remove memory size local var if not needed '''
	@property
	def memory_size(self):
		return self._memory_size

	@memory_size.setter
	def memory_size(self, value):

		if value > 0 and value < self._memory_size:
			self.memory = self.memory[:value]
		self._memory_size = value

	def remember(self, current_state, a, r, next_state, game_over):

		self.input_shape = current_state.shape[1:] #store the shape of the input states for later use
		history = np.concatenate([current_state.flatten(), np.array(a).flatten(), np.array(r).flatten(), next_state.flatten(), 1 * np.array(game_over).flatten()])
		self.memory.append(history)
        
		if self.memory_size > 0 and len(self.memory) > self.memory_size:
			self.memory.pop(0)

	def reset_memory(self):
		self.memory = []

	def get_batch(self, model, batch_size, gamma=0.9):
	    
		#for initial stages of the game when there isnt enough histories in the memory
		if len(self.memory) < batch_size:
			batch_size = len(self.memory)

		#set up the config details
		num_actions = model.output_shape[-1]
		samples = np.array(sample(self.memory, batch_size)) # sample a number of histories equal to the batch size
		input_dim = np.prod(self.input_shape)

		#extract the current state, action, reward, next state and game over flag from the sampled data
		current_states = samples[:, 0 : input_dim]
		actions = samples[:, input_dim]
		rewards = samples[:, input_dim + 1]
		next_states = samples[:, input_dim + 2 : 2 * input_dim + 2]
		game_overs = samples[:, 2 * input_dim + 2]

		#pdb.set_trace()
		rewards = rewards.repeat(num_actions).reshape((batch_size, num_actions))
		game_overs = game_overs.repeat(num_actions).reshape((batch_size, num_actions))
		current_states = current_states.reshape((batch_size, ) + self.input_shape)
		next_states = next_states.reshape((batch_size, ) + self.input_shape)

		#do the forward pass on the current and next states
		features = np.concatenate([current_states, next_states], axis=0)
		predictions = model.predict(features)

		#perform the q-function update
		optimal_q_values = np.max(predictions[batch_size:], axis=1).repeat(num_actions).reshape((batch_size, num_actions))
		delta = np.zeros((batch_size, num_actions))
		actions = actions.astype("int")
		delta[np.arange(batch_size), actions] = 1
		targets = (1 - delta) * predictions[:batch_size] + delta * (rewards + (1 - game_overs) * optimal_q_values)

		return current_states, targets
