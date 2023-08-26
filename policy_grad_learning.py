from __future__ import print_function
__author__ = "Abhimanyu Banerjee"

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Input
import random
#from keras.layers import Convolution2D, MaxPooling2D
#from keras.utils import np_utils
#from keras import backend as K
import pdb

from games import tttAgent2D
from games.board import empty_state, is_game_over

def modelSetup(I, H, L):
	model = Sequential()
	model.add(Dense(H, input_dim=I, activation="relu"))
	
	for layer in range(L-1):
		model.add(Dense(H, activation="relu"))

	model.add(Dense(output_dim=1, activation="sigmoid"))
	#model.compile(loss='mean_squared_error', optimizer='sgd')

	print(model.summary())

	return model

def policy_forward(model, input):
	
	activations = []
	y = model.predict(input) #forward pass

	for layer in model.layers:
		pdb.set_trace()
		activations.append(layer.output)
	
	return activations[:-1], y

def policy_backward(eph, epdlogp):
	
	''' backward pass. (eph is array of intermediate hidden states) '''
	pass
	
def play_episode(agent, model, state):

	''' complete one episode of play while collecting the input states, the hidden state
	activations and the output losses'''

	states, activations, rewards, losses = [], [], [], []
	winner = None
	
	while winner != 0:
		
		input = np.transpose(np.vstack([cell for row in state for cell in row]))

		#forward the policy and sample an action from the return probability
		activation, action_prob = policy_forward(model, input)
		action = np.around(action_prob*board_shape[0]*board_shape[1])

		#record the intermediate values for use during backprop later
		states.appened(state)
		activations.append(activation)
		losses.append(action - action_prob)

		#play the next move
		state, reward = agent.play_opponent(state, action)
		rewards.append(reward)
		winner = is_game_over(state)

	return states, activations, rewards, losses

if __name__ == "__main__":

	# hyperparameters
	board_size = 3
	hidden_layer_neurons = 200 # number of hidden layer neurons
	hidden_layers = 1
	batch_size = 10 # every how many episodes to do a param update?
	learning_rate = 1e-4
	gamma = 0.99 # discount factor for reward
	decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2

	input_size = board_size ** 2
	model = modelSetup(input_size, hidden_layer_neurons, hidden_layers)
	AI = tttAgent2D(symbol=1, behaviour_threshold=0.8, is_learning=False)

	#setup buffers for the gradients and rmsprop
	batch_grads, batch_rms = {}, {}
	idx = 0
	for layer in model.layers:
		batch_grads[idx] = np.zeros_like(layer.get_weights()[0])
		batch_rms[idx] = np.zeros_like(layer.get_weights()[0])
		idx += 1

	while True:

		states, activations, rewards, losses = play_episode(AI, model, empty_state())
		episodes += 1 #update counter for episode

		episode_inputs = np.vstack(states)
		#todo: figure out a good data structure for episode activations
		episode_losses = np.vstack(losses)
		episode_rewards = discount_rewards(np.vstack(rewards))

		#standardize the discounted episode rewards
		episode_rewards -= np.mean(episode_rewards)
		episode_rewards /= np.std(episode_rewards)

		episode_losses *= episode_rewards
		grads = policy_backward(episode_activations, episode_losses) # get accumulated gradients over one episode

		#accumulate gradients over batches
		for idx in range(len(model.layers)):
			batch_grads[idx] += grads[idx]

		if episodes % batch_size == 0:

			#update weights for each layer in model
			for idx in range(len(model.layers)):
				gradient = batch_grads[idx]
				batch_rms[idx] = decay_rate * batch_rms[idx] + (1 - decay_rate) * gradient**2
				updated_weights = learning_rate * gradient / (np.sqrt(batch_rms[idx]) + 1e-5)
				model.layers[idx].set_weights(updated_weights)

