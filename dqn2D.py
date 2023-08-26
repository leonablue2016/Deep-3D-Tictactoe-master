from __future__ import print_function
__author__ = "Abhimanyu Banerjee"

import numpy as np
np.random.seed(1337)  # for reproducibility

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy

from random import choice
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.layers import Convolution2D, MaxPooling2D
#from keras.utils import np_utils
#from keras import backend as K
import pdb

from memory import ExperienceReplay

from agents import TDAgent
from games.board import empty_state, is_game_over, flatten_state, open_spots, print_board

class DQNAgent(object):

	def __init__(self, model, memory=None, memory_size=1000):
		if memory:
			self.memory = memory
		else:
			self.memory = ExperienceReplay(memory_size)

		self.model = model

	''', epsilon_rate=0.5'''
	def train_network(self, game, num_epochs=1000, batch_size=50, gamma=0.9, epsilon=[.1, 1], epsilon_rate=0.5, reset_memory=False, observe=2):

		model = self.model
		game.reset_agent()
		nb_actions = model.output_shape[-1]
		win_count, loss_count, total_reward, total_q = 0, 0, 0.0, 0.0
		batch_probs, avg_reward, avg_q = [], [], []
		total_q, total_reward = 0.0, 0.0 
		delta = (epsilon[1] - epsilon[0]) /(num_epochs*epsilon_rate)
		epsilon_set = np.arange(epsilon[0], epsilon[1], delta)
		
		for epoch in range(1, num_epochs+1):
			
			loss, winner = 0., None
			num_plies, game_q, game_reward = 0, 0.0, 0.0
			current_state = empty_state(game.dims) #reset game
			game_over = False
			#self.clear_frames()
			
			if reset_memory:
				self.memory.reset_memory()
			
			if epoch % (num_epochs/1000) == 0:
				batch_probs.append(self.measure_performance(game, 100))
			
			while not game_over:

				#pdb.set_trace()
				if np.random.random() > epsilon_set[int(epoch*epsilon_rate - 0.5)]: #or epoch < observe:
					empty_cells = open_spots(current_state)
					move = choice(empty_cells) # choose move randomly from available moves
				
				#choose the action for which Q function gives max value
				else: 
					q = model.predict(flatten_state(current_state))
					move = int(np.argmax(q[0]))
					game_q += np.amax(q[0])


				next_state, reward = game.play_board(deepcopy(current_state), move)

				game_reward += reward
				num_plies += 1

				#check who, if anyone, has won
				if reward != 0 or len(open_spots(next_state)) == 0:
					game_over = True

				'''reward,'''
				transition = [flatten_state(current_state), move, reward, flatten_state(next_state), game_over] 
				self.memory.remember(*transition)
				current_state = next_state #update board state
				
				if epoch % observe == 0:
					
					batch = self.memory.get_batch(model=model, batch_size=batch_size, gamma=gamma)
					if batch:
						inputs, targets = batch
						loss += float(model.train_on_batch(inputs, targets))
				
				'''if checkpoint and ((epoch + 1 - observe) % checkpoint == 0 or epoch + 1 == num_epochs):
					model.save_weights('weights.dat')'''
			
			if reward == -1*game.symbol: #ttt agent's symbol is inverted to get the model's symbol
				win_count += 1
			
			total_q += game_q / num_plies
			total_reward += game_reward / num_plies
			avg_q.append(total_q / epoch)
			avg_reward.append(total_reward / epoch)
			print("Epoch {:03d}/{:03d} | Loss {:.4f} | Average Q {:.2f} | Average Reward {:.2f} | Wins {}".format(epoch, num_epochs, loss, avg_q[epoch-1], avg_reward[epoch-1], win_count))

		game_epochs = [i for i in range(1000)]
		win_probs = [probs[2] for probs in batch_probs]
		
		plt.plot(game_epochs, win_probs, label="Win Probability", color="g")
		#plt.plot(epochs, avg_reward, label="Average Reward", color="r")
		plt.ylim((0,1.5))
		plt.xlabel('Epochs')
		plt.ylabel('Probability')
		#plt.ylabel('Average Reward')

		#pdb.set_trace()
		epochs = [i for i in range(num_epochs)]
		plt.title('DQN Agent Performance v/s Q-Learning Agent (epsilon-rate={0})'.format(epsilon_rate))
		plt.legend()
		plt.savefig('dqn_epsilon-rate={0}.png'.format(epsilon_rate))
		plt.close()

		plt.plot(epochs, avg_q, label="Average Q Value", color="b")
		plt.xlabel("Epochs")
		plt.ylabel("Q Value")
		plt.title('DQN Agent Performance v/s Q-Learning Agent (epsilon-rate={0})'.format(epsilon_rate))
		plt.legend()
		plt.savefig('dqn_avg_q.png')
		plt.close()

		plt.plot(epochs, avg_reward, label="Average Reward", color="b")
		plt.xlabel("Epochs")
		plt.ylabel("Reward")
		plt.title('DQN Agent Performance v/s Q-Learning Agent (epsilon-rate={0})'.format(epsilon_rate))
		plt.legend()
		plt.savefig('dqn_avg_reward.png')
		plt.close()

		model.save_weights('weights.dat') # save model weights

		#final set of games to evaluate agent's performance after learning
		final_stats = self.measure_performance(game, 100)
		return np.array(win_probs).sum(), final_stats


	def play_game(self, game):
			
		#set up game config
		model = self.model
		current_state = empty_state(game.dims) #reset game
		random_counter = 3 # number of random moves allowed in a game

		while True:
			#pdb.set_trace()
			q_values = model.predict(flatten_state(current_state))
			move = int(np.argmax(q_values[0]))

			next_state, reward = game.play_board(deepcopy(current_state), move)
			
			if np.array_equal(current_state, next_state):
				if random_counter > 0:
					empty_cells = open_spots(current_state)
					move = choice(empty_cells) # choose move randomly from available moves
					next_state, reward = game.play_board(deepcopy(current_state), move)
					random_counter -= 1
				else:
					return -1

			#check who, if anyone, has won
			if reward != 0 or len(open_spots(next_state)) == 0:
				return reward

			current_state = next_state

	def play_human(self, game_dims):

		model = self.model
		model.load_weights("weights.dat")
		state = empty_state(game_dims) #reset game
		random_counter = 3 # number of random moves allowed in a game
		turn = 1

		print("\nBoard at turn {}".format(turn))
		print_board(state)
		while True:
			
			human_move = int(input("Enter the cell you want to mark (0-15):"))
			row, col = int(human_move / len(state)), human_move % len(state) #get the row and column to mark from the chosen action
			state[row, col] = 1
			turn += 1
			
			print("\nBoard at turn {}".format(turn))
			print_board(state)

			winner = is_game_over(state)
			print("Winner is ", winner)
			if winner == 1:
				print("Congratulations! You have Won!")
				break
			elif winner == -1:
				print("Bummer! You lost to the AI!!")
				break

			q_values = model.predict(flatten_state(state))
			computer_move = int(np.argmax(q_values[0]))
			new_state = deepcopy(state)
			row, col = int(computer_move / len(state)), computer_move % len(state) #get the row and column to mark from the chosen action
			new_state[row, col] = -1
			if np.array_equal(new_state, state):
				if random_counter > 0:
					empty_cells = open_spots(state)
					computer_move = choice(empty_cells) # choose move randomly from available moves
					row, col = int(computer_move / len(state)), computer_move % len(state) #get the row and column to mark from the chosen action
					new_state[row, col] = -1
					random_counter -= 1

				else:
					print("D'oh! The AI has much to learn still! Sorry for wasting your time, hooman!")
					break
			state = new_state
			turn += 1

			print("\nBoard at turn {}".format(turn))
			print_board(state)

			winner = is_game_over(state)
			print("Winner is ", winner)
			if winner == 1:
				print("Congratulations! You have Won!")
				break
			elif winner == -1:
				print("Bummer! You lost to the AI!!")
				break

			if len(open_spots(state)) == 0:
				print("Alas! The Game has been drawn!")
				break

	def measure_performance(self, game, num_games):
		
		probs, games_played = [0,0,0], 0

		for i in range(num_games):
			
			#print("Starting Game {}".format(i+1))
			winner = self.play_game(game)
			
			if winner != -1:
			
				games_played += 1	
				if winner == 0:
					probs[1] += 1.0
				elif winner == 1:
					probs[2] += 1.0
				else:
					probs[0] += 1.0
			#print("Ending Game {}".format(i+1))
			
		if games_played > 0:
			probs[0] = probs[0] * 1. / games_played
			probs[1] = probs[1] * 1. / games_played
			probs[2] = probs[2] * 1. / games_played
		
		return probs 

def modelSetup(I, H, L):
	
	model = Sequential()
	model.add(Dense(H, input_dim=I, activation="relu"))
	
	for layer in range(L-1):
		model.add(Dense(H, activation="relu"))

	model.add(Dense(output_dim=I, activation="sigmoid"))
	model.compile(loss='mean_squared_error', optimizer='sgd')

	print(model.summary())

	return model

if __name__ == "__main__":

	# hyperparameters
	board_size = 4
	hidden_layer_neurons = 512 # number of hidden layer neurons
	hidden_layers = 15
	batch_size = 16 # every how many episodes to do a param update?
	learning_rate = 1e-4
	gamma = 0.99 # discount factor for reward
	decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
	memory_size = 10000 # size of the replay memory

	input_size = board_size ** 2
	model = modelSetup(input_size, hidden_layer_neurons, hidden_layers)
	dqnAgent = DQNAgent(model=model, memory_size=memory_size)

	'''tdAgent2D = TDAgent(symbol=-1, is_learning=False, dims=2, build_states=False)
	tdAgent2D.reset_agent()

	cum_wins, final_stats = 0.0, []

	dqnAgent = DQNAgent(model=model, memory_size=memory_size)
	#win_ratios.append(dqnAgent.train_network(tictactoe, batch_size=batch_size, num_epochs=10000, gamma=gamma))
	#win_ratios.append(dqnAgent.train_network(tictactoe, batch_size=batch_size, num_epochs=10000, gamma=gamma, epsilon_rate=1))
	#win_ratios.append(dqnAgent.train_network(tictactoe, batch_size=batch_size, num_epochs=10000, gamma=gamma, epsilon_rate=0.8))
	cum_wins, final_stats = dqnAgent.train_network(tdAgent2D, batch_size=batch_size, num_epochs=1000, gamma=gamma, epsilon_rate=0.3)

	print("\nTotal cumulative wins with heuristic: ", cum_wins)
	print("\n Final Matchup Results with heuristic: {0} Wins | {1} Draws | {2} Losses".format(final_stats[2], final_stats[1], final_stats[0]))
	print("\nArchitecture Details:")
	print("Batch size {0} | Hidden Layers: {1} | Memory size: {2}".format(batch_size, hidden_layers, memory_size))
'''
	dqnAgent.play_human(2)
