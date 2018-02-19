#!/usr/bin/env python

import os
import sys
import h5py
import math
import random
import numpy as np
from scipy.signal import savgol_filter
import Constants as const
from collections import deque
import matplotlib.pyplot as plt
import matplotlib as mpl

"""
Utility class for static methods
"""

class Utility:
	"""
	Directory generation methods
	"""
	@staticmethod
	def getHDF5DataDir():
		return os.path.join(const.BASE_DIR, const.DATA_DIR_HDF5)
	@staticmethod
	def getTensorboardDir():
		return os.path.join(const.BASE_DIR, const.TENSORBOARD_DIR)
	@staticmethod
	def getModelDir():
		filename = "{}.tflearn".format(const.MODEL_NAME)
		return os.path.join(const.BASE_DIR, const.MODELS_DIR, filename)
	@staticmethod
	def getVideoDir():
		return os.path.join(const.BASE_DIR, const.VIDEO_DIR)

	"""
	ICIP 2018 directory methods
	"""
	@staticmethod
	def getICIPDataDir():
		return os.path.join(const.BASE_DIR, const.ICIP_DATA_DIR)
	@staticmethod
	def getICIPFigureDir():
		return os.path.join(const.BASE_DIR, const.ICIP_FIGURE_DIR)
	@staticmethod
	def getICIPModelDir():
		return os.path.join(const.BASE_DIR, const.ICIP_MODELS_DIR)
	@staticmethod
	def getICIPTensorboardDir():
		return os.path.join(const.BASE_DIR, const.ICIP_TENSORBOARD_DIR)

	"""
	General utility functions
	"""

	# Compute the shortest path action sequence from a -> b
	@staticmethod
	def actionSequenceBetweenCoordinates(a_x, a_y, b_x, b_y):
		actions = []

		# Loop until we're at the ending position
		while (a_x, a_y) != (b_x, b_y):
			# Find possible actions for the current-end relative vector
			possible_actions = Utility.possibleActionsForAngle(a_x, a_y, b_x, b_y)

			# Randomly select a possible action (for 45 degree cases)
			rand_idx = random.randint(0, len(possible_actions)-1)
			choice = possible_actions[rand_idx]

			# Perform the chosen action
			if choice == 'F': 	a_y -= const.MOVE_DIST
			elif choice == 'B': a_y += const.MOVE_DIST
			elif choice == 'L': a_x -= const.MOVE_DIST
			elif choice == 'R': a_x += const.MOVE_DIST
			else: Utility.die("Action: {} not recognised!".format(choice), __file__)

			# Store the chosen action in the list of actions
			actions.append(choice)

		return actions

	# Converts from a single action to a class vector required by the dnn model
	# e.g. 'F' -> [1,0,0,0]
	@staticmethod
	def actionToClassVector(action):
		if const.USE_EXT_ACTIONS:
			vec = np.zeros(len(const.EXT_ACTIONS))
		else:
			vec = np.zeros(len(const.ACTIONS))

		if action == 'F': vec[0] = 1
		elif action == 'B': vec[1] = 1
		elif action == 'L': vec[2] = 1
		elif action == 'R': vec[3] = 1
		elif action == 'N' and const.USE_EXT_ACTIONS: vec[4] = 1
		else: Utility.die("Action not recognised or extended actions not enabled", __file__)

		return vec

	# The opposite of the above function
	@staticmethod
	def classVectorToAction(class_vec):
		action = ''

		if class_vec[0]: action = 'F'
		elif class_vec[1]: action = 'B'
		elif class_vec[2]: action = 'L'
		elif class_vec[3]: action = 'R'
		elif const.USE_EXT_ACTIONS and class_vec[4]: action = 'N'
		else: Utility.die("Action not recognised or extended actions not enabled.", __file__)

		return action

	# Given the position of a target, find the angle between the agent position and
	# the target and choose the best possible action towards navigating towards that
	# target object
	@staticmethod
	def bestActionForAngle(a_x, a_y, b_x, b_y):
		# Compute angle between given points
		angle = Utility.angleBetweenPoints(a_x, a_y, b_x, b_y)

		if angle < math.pi/4 and angle > -math.pi/4: action = 'F'
		elif angle >= math.pi/4 and angle < 3*math.pi/4: action = 'L'
		elif angle <= math.pi/4 and angle > -3*math.pi/4: action = 'R'
		elif angle >= 3*math.pi/4 or angle <= -3*math.pi/4: action = 'B'
		else: Utility.die("Angle is not in [0,360] degrees", __file__)

		# Make sure the assigned action is valid
		assert(action in const.ACTIONS)

		return action

	# Given a position and the map's boundaries, return a list of possible
	# actions that don't result in the agent going out of bounds
	@staticmethod
	def possibleActionsForPosition(x, y):
		# Get the list of all actions
		actions = list(const.ACTIONS)

		# Check map boundaries in x axis
		if x == 0: actions.remove('L')
		elif x == const.MAP_WIDTH - 1: actions.remove('R')

		# Check map boundaries in y axis
		if y == 0: actions.remove('F')
		elif y == const.MAP_HEIGHT - 1: actions.remove('B')

		return actions

	# Very similar to "bestActionForAngle" except for the case when an angle is 45 degrees
	# it returns both F, R in a char vector
	@staticmethod
	def possibleActionsForAngle(a_x, a_y, b_x, b_y):
		# Compute angle between given points
		angle = Utility.angleBetweenPoints(a_x, a_y, b_x, b_y)

		# If the angle is exactly diagonal (in 45 degree increments)
		# top left 
		if angle == math.pi/4: return ['F', 'L']
		elif angle == 3*math.pi/4: return ['L', 'B']
		elif angle == -3*math.pi/4: return ['B', 'R']
		elif angle == -math.pi/4: return ['R', 'F']
		else: return Utility.bestActionForAngle(a_x, a_y, b_x, b_y) 

	@staticmethod
	def angleBetweenPoints(a_x, a_y, b_x, b_y):
		# Get relative position
		rel_x = a_x - b_x
		rel_y = a_y - b_y

		# Compute angle
		angle = math.atan2(rel_x, rel_y)

		# print "Angle = {} for point ({},{})".format(math.degrees(angle), rel_x, rel_y)
	
		return angle

	# Rotate or shift sequence by n
	@staticmethod
	def rotateList(sequence, n):
		return sequence[n:] + sequence[:n]

	# Returns the Euclidean distance between input coordinates a, b in tuple form (x, y)
	@staticmethod
	def distanceBetweenPoints(a, b):
		return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

	# Given the current state of the occupancy map, extract the x,y grid coordinates
	# of the agent and ensure there's only one
	@staticmethod
	def getAgentCoordinatesFromMap(occupancy_map):
		# Find the current agent position
		if const.OCCUPANCY_MAP_MODE == const.VISITATION_MODE:
			index = np.where(occupancy_map == const.AGENT_VAL)
		elif const.OCCUPANCY_MAP_MODE == const.MOTION_MODE:
			index = np.where(occupancy_map[:,:,1] == const.MOTION_HIGH_VALUE)
		else:
			Utility.die("Occupancy map mode not recognised", __file__)

		# Ensure we only found one position
		if index[0].shape[0] > 1 and index[1].shape[0] > 1:
			Utility.die("Found more than one agent location!", __file__)

		return index[1][0], index[0][0]

	# Check whether the supplied position is out of map bounds
	@staticmethod
	def checkPositionInBounds(x, y):
		if x < 0 or y < 0 or x >= const.MAP_WIDTH or y >= const.MAP_HEIGHT:
			return False

		return True

	@staticmethod
	def die(message, file):
		print "\nERROR MESSAGE:_________________\n\"{}\"\nin file: {}\nExiting..".\
			format(message, file)
		sys.exit(0)

	# Method takes two h5 databases of equal dimensions and combines them into a single file
	@staticmethod
	def combineH5Databases(out_path, file1_path, file2_path):
		f1 = h5py.File(file1_path, 'r')
		f2 = h5py.File(file2_path, 'r')

		# Extract datasets from both
		f1_X0 = f1['X0'][()]
		f1_X1 = f1['X1'][()]
		f1_Y = f1['Y'][()]
		f2_X0 = f2['X0'][()]
		f2_X1 = f2['X1'][()]
		f2_Y = f2['Y'][()]

		# Check the dataset shapes agree
		assert(f1_X0.shape[1:] == f2_X0.shape[1:])
		assert(f1_X1.shape[1:] == f2_X1.shape[1:])
		assert(f1_Y.shape[1:] == f2_Y.shape[1:])

		# Append to each other
		X0 = np.concatenate((f1_X0, f2_X0), axis=0)
		X1 = np.concatenate((f1_X1, f2_X1), axis=0)
		Y = np.concatenate((f1_Y, f2_Y), axis=0)

		# Open the new dataset file (WARNING: will overwrite exisiting file!)
		out = h5py.File(out_path, 'w')

		# Create the datasets
		out.create_dataset('X0', data=X0)
		out.create_dataset('X1', data=X1)
		out.create_dataset('Y', data=Y)

		# Finish up
		out.close()

	"""
	Graph drawing utility methods for evaluating algorithm performance
	"""

	@staticmethod
	def drawGenerationTimeGraph(t_s, t_c, s_t, e_t):
		plt.style.use('seaborn-darkgrid')
		# Construct size of targets vector
		T = np.arange(s_t, e_t)

		# Plot vectors
		plt.plot(T, np.average(t_s, axis=1), label="Sequence")
		plt.plot(T, np.average(t_c, axis=1), label="Closest")

		# Graph parameters
		plt.xlabel('|R|')
		plt.ylabel('time(s)')
		plt.legend(loc="center right")
		plt.show()

	@staticmethod
	def drawGenerationLengthGraph(m_s, m_c, s_t, e_t):
		plt.style.use('seaborn-darkgrid')
		# Construct size of targets vector
		T = np.arange(s_t, e_t)

		seq_avg = np.average(m_s, axis=1)
		clo_avg = np.average(m_c, axis=1)

		# print seq_avg
		# print clo_avg

		hist_vec = (m_c - m_s).flatten()

		# print hist_vec

		N, bins, patches = plt.hist(hist_vec, bins=13, normed=True, histtype='stepfilled',)

		# Plot vectors
		# plt.plot(T, seq_avg, label="Sequence")
		# plt.plot(T, clo_avg, label="Closest")

		# Graph parameters
		plt.xlabel('Difference in solution length')
		# plt.ylabel('moves')
		plt.legend(loc="center right")
		plt.tight_layout()
		plt.savefig("{}/solution-generation-hist.pdf".format(Utility.getICIPFigureDir()))
		plt.show()

	@staticmethod
	def drawGenerationGraphs(m_s, m_c, t_s, t_c, num_targets):
		plt.style.use('seaborn-darkgrid')
		fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True)

		# Number of targets array
		R = np.asarray(num_targets)

		# Average results
		m_seq_avg = np.average(m_s, axis=1)
		m_clo_avg = np.average(m_c, axis=1)
		t_seq_avg = np.average(t_s, axis=1)
		t_clo_avg = np.average(t_c, axis=1)

		axs[0].plot(R, m_seq_avg, 'b', label="Target Ordering")
		axs[0].plot(R, m_clo_avg, 'r', label="Closest Unvisited")
		axs[1].plot(R, t_seq_avg, 'b', label="Target Ordering")
		axs[1].plot(R, t_clo_avg, 'r', label="Closest Unvisited")

		axs[0].set_ylabel("|Moves|")
		axs[0].set_xlabel("|R|")
		axs[1].set_ylabel("Time (s)")
		axs[1].set_xlabel("|R|")

		plt.legend(loc="upper left")
		plt.tight_layout()
		plt.savefig("{}/solution-generation.pdf".format(Utility.getICIPFigureDir()))
		plt.show()

	# Draws graph of training data instance size versus best validation accuracy
	@staticmethod
	def drawDatasetSizeAccuracyGraph():
		# Load the data
		base = Utility.getICIPDataDir()
		val_5k = np.genfromtxt("{}/5k_val_acc.csv".format(base), delimiter=',', skip_header=1, names=['x', 'y', 'z'])
		val_10k = np.genfromtxt("{}/10k_val_acc.csv".format(base), delimiter=',', skip_header=1, names=['x', 'y', 'z'])
		val_20k = np.genfromtxt("{}/20k_val_acc.csv".format(base), delimiter=',', skip_header=1, names=['x', 'y', 'z'])
		val_40k = np.genfromtxt("{}/40k_val_acc.csv".format(base), delimiter=',', skip_header=1, names=['x', 'y', 'z'])
		val_60k = np.genfromtxt("{}/60k_val_acc.csv".format(base), delimiter=',', skip_header=1, names=['x', 'y', 'z'])

		# Extract maximum validation values
		max_5 = val_5k['z'].max()
		max_10 = val_10k['z'].max()
		max_20 = val_20k['z'].max()
		max_40 = val_40k['z'].max()
		max_60 = val_60k['z'].max()

		x = np.asarray([5000, 10000, 20000, 40000, 60000])
		y = np.asarray([max_5, max_10, max_20, max_40, max_60])

		plt.style.use('seaborn-darkgrid')
		plt.plot(x, y)
		plt.xlabel('Dataset size')
		plt.ylabel('Accuracy')
		plt.tight_layout()
		plt.savefig("{}/dataset-size-accuracy.pdf".format(Utility.getICIPFigureDir()))
		plt.show()

	# Method for drawing a graph that compares training and validation accuracy versus
	# epochs.
	@staticmethod
	def drawAccuracyGraph():
		# Convolutional based smoothing function
		def smooth(y, box_pts):
		    return savgol_filter(y, box_pts, 3)
		    # return y_smooth

		# Load the data
		base = Utility.getICIPDataDir()
		acc_5k = np.genfromtxt("{}/5k_train_acc.csv".format(base), delimiter=',', skip_header=1, names=['x', 'y', 'z'])
		val_5k = np.genfromtxt("{}/5k_val_acc.csv".format(base), delimiter=',', skip_header=1, names=['x', 'y', 'z'])
		acc_10k = np.genfromtxt("{}/10k_train_acc.csv".format(base), delimiter=',', skip_header=1, names=['x', 'y', 'z'])
		val_10k = np.genfromtxt("{}/10k_val_acc.csv".format(base), delimiter=',', skip_header=1, names=['x', 'y', 'z'])
		acc_20k = np.genfromtxt("{}/20k_train_acc.csv".format(base), delimiter=',', skip_header=1, names=['x', 'y', 'z'])
		val_20k = np.genfromtxt("{}/20k_val_acc.csv".format(base), delimiter=',', skip_header=1, names=['x', 'y', 'z'])
		acc_40k = np.genfromtxt("{}/40k_train_acc.csv".format(base), delimiter=',', skip_header=1, names=['x', 'y', 'z'])
		val_40k = np.genfromtxt("{}/40k_val_acc.csv".format(base), delimiter=',', skip_header=1, names=['x', 'y', 'z'])
		acc_60k = np.genfromtxt("{}/60k_train_acc.csv".format(base), delimiter=',', skip_header=1, names=['x', 'y', 'z'])
		val_60k = np.genfromtxt("{}/60k_val_acc.csv".format(base), delimiter=',', skip_header=1, names=['x', 'y', 'z'])

		# Define the style and subplots
		plt.style.use('seaborn-darkgrid')
		fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

		# Scale x-axis values down to the number of epochs
		acc_5k['y'] = (const.NUM_EPOCHS*(acc_5k['y'] - acc_5k['y'].min())) / acc_5k['y'].max()
		acc_10k['y'] = (const.NUM_EPOCHS*(acc_10k['y'] - acc_10k['y'].min())) / acc_10k['y'].max()
		acc_20k['y'] = (const.NUM_EPOCHS*(acc_20k['y'] - acc_20k['y'].min())) / acc_20k['y'].max()
		acc_40k['y'] = (const.NUM_EPOCHS*(acc_40k['y'] - acc_40k['y'].min())) / acc_40k['y'].max()
		acc_60k['y'] = (const.NUM_EPOCHS*(acc_60k['y'] - acc_60k['y'].min())) / acc_60k['y'].max()

		val_5k['y'] = ((const.NUM_EPOCHS*(val_5k['y'] - val_5k['y'].min())) / val_5k['y'].max()) + 1
		val_10k['y'] = ((const.NUM_EPOCHS*(val_10k['y'] - val_10k['y'].min())) / val_10k['y'].max()) + 1
		val_20k['y'] = ((const.NUM_EPOCHS*(val_20k['y'] - val_20k['y'].min())) / val_20k['y'].max()) + 1
		val_40k['y'] = ((const.NUM_EPOCHS*(val_40k['y'] - val_40k['y'].min())) / val_40k['y'].max()) + 1 
		val_60k['y'] = ((const.NUM_EPOCHS*(val_60k['y'] - val_60k['y'].min())) / val_60k['y'].max()) + 1

		# Alpha constant
		alpha = 0.2

		# Smoothing parameter
		tra_s = 99

		# Plot to training accuracy graph
		axs[0].plot(acc_5k['y'], acc_5k['z'], color='y', alpha=alpha)
		axs[0].plot(acc_10k['y'], acc_10k['z'], color='k', alpha=alpha)
		axs[0].plot(acc_20k['y'], acc_20k['z'], color='r', alpha=alpha)
		axs[0].plot(acc_40k['y'], acc_40k['z'], color='g', alpha=alpha)
		axs[0].plot(acc_60k['y'], acc_60k['z'], color='b', alpha=alpha)

		axs[0].plot(acc_5k['y'], smooth(acc_5k['z'], tra_s), color='y', label='5k')
		axs[0].plot(acc_10k['y'], smooth(acc_10k['z'], tra_s), color='k', label='10k')
		axs[0].plot(acc_20k['y'], smooth(acc_20k['z'], tra_s), color='r', label='20k')
		axs[0].plot(acc_40k['y'], smooth(acc_40k['z'], tra_s), color='g', label='40k')
		axs[0].plot(acc_60k['y'], smooth(acc_60k['z'], tra_s), color='b', label='60k')

		# Plot to validation accuracy graph
		axs[1].plot(val_5k['y'], val_5k['z'], color='y', alpha=alpha)
		axs[1].plot(val_10k['y'], val_10k['z'], color='k', alpha=alpha)
		axs[1].plot(val_20k['y'], val_20k['z'], color='r', alpha=alpha)
		axs[1].plot(val_40k['y'], val_40k['z'], color='g', alpha=alpha)
		axs[1].plot(val_60k['y'], val_60k['z'], color='b', alpha=alpha)

		# Smoothing constant
		val_s = 11

		axs[1].plot(val_5k['y'], smooth(val_5k['z'], val_s), color='y', label='5k')
		axs[1].plot(val_10k['y'], smooth(val_10k['z'], val_s), color='k', label='10k')
		axs[1].plot(val_20k['y'], smooth(val_20k['z'], val_s), color='r', label='20k')
		axs[1].plot(val_40k['y'], smooth(val_40k['z'], val_s), color='g', label='40k')
		axs[1].plot(val_60k['y'], smooth(val_60k['z'], val_s), color='b', label='60k')

		# Set axis labels for subplots
		axs[0].set_xlabel("Epochs")
		axs[0].set_ylabel("Accuracy")
		axs[0].set_title("Training")
		axs[1].set_xlabel("Epochs")
		axs[1].set_title("Validation")

		plt.axis([0, 50, 0.5, 0.9])
		plt.legend(loc="upper right")
		plt.tight_layout()

		plt.savefig("{}/motion-training-accuracy.pdf".format(Utility.getICIPFigureDir()))
		plt.show()

	@staticmethod
	def drawModelLengthHistogram():
		base = Utility.getICIPDataDir()
		data_seq = np.load("{}/test_data_seq.npy".format(base))
		data_clo = np.load("{}/test_data_clo.npy".format(base))
		data_nav = np.load("{}/test_data_NAIVE.npy".format(base))
		data_seq_sim = np.load("{}/test_data_SIMULATOR_seq.npy".format(base))
		#data_seq_sim = np.load("{}/test_data_GAUS_SEQ.npy".format(base))

		assert(data_seq.shape == data_clo.shape)
		assert(data_clo.shape == data_nav.shape)
		assert(data_seq_sim.shape == data_nav.shape)

		num_inst = data_seq_sim.shape[0]

		# Find stats about how many times loop detection was triggered
		loop_seq = np.where(data_seq[:,2] > 0)[0].shape[0]
		loop_clo = np.where(data_clo[:,2] > 0)[0].shape[0]
		loop_seq_sim = np.where(data_seq_sim[:,2] > 0)[0].shape[0]
		print "__Loop detected________________"
		print "TO: {}%".format((float(loop_seq)/num_inst)*100)
		print "CU: {}%".format((float(loop_clo)/num_inst)*100)
		print "TO+S: {}%".format((float(loop_seq_sim)/num_inst)*100)
		print "\n\n"

		# Find stats about how many times loop detection is triggered again once it 
		# has already been triggered
		doub_seq = np.where(data_seq[:,2] > 1)[0].shape[0]
		doub_clo = np.where(data_clo[:,2] > 1)[0].shape[0]
		doub_seq_sim = np.where(data_seq_sim[:,2] > 1)[0].shape[0]
		print "__Multiple Loops detected________________"
		print "TO: {}%".format((float(doub_seq)/loop_seq)*100)
		print "CU: {}%".format((float(doub_clo)/loop_clo)*100)
		print "TO+S: {}%".format((float(doub_seq_sim)/loop_seq_sim)*100)
		print "\n"

		# Find stats about % of time model generates over 100 moves
		over_seq = np.where(data_seq[:,0] > 100)[0].shape[0]
		over_clo = np.where(data_clo[:,0] > 100)[0].shape[0]
		over_nav = np.where(data_nav[:,0] > 100)[0].shape[0]
		over_seq_sim = np.where(data_seq_sim[:,0] > 100)[0].shape[0]
		print "__Over 100 moves________________"
		print "TO: {}%".format((float(over_seq)/num_inst)*100)
		print "CU: {}%".format((float(over_clo)/num_inst)*100)
		print "NS: {}%".format((float(over_nav)/num_inst)*100)
		print "TO+S: {}%".format((float(over_seq_sim)/num_inst)*100)
		print "\n"

		hist_vec_seq = data_seq[:,0] - data_seq[:,1]
		hist_vec_clo = data_clo[:,0] - data_clo[:,1]
		hist_vec_nav = data_nav[:,0] - data_nav[:,1]
		hist_vec_seq_sim = data_seq_sim[:,0] - data_seq_sim[:,1]

		# Find stats about % of time model generates globally-optimal solution
		opt_seq = np.where(hist_vec_seq == 0)[0].shape[0]
		opt_clo = np.where(hist_vec_clo == 0)[0].shape[0]
		opt_nav = np.where(hist_vec_nav == 0)[0].shape[0]
		opt_seq_sim = np.where(hist_vec_seq_sim == 0)[0].shape[0]
		print "__Globally Optimal?________________"
		print "TO: {}%".format((float(opt_seq)/num_inst)*100)
		print "CU: {}%".format((float(opt_clo)/num_inst)*100)
		print "NS: {}%".format((float(opt_nav)/num_inst)*100)
		print "TO+S: {}%".format((float(opt_seq_sim)/num_inst)*100)
		print "\n"

		# Find stats about % of time model generates <10 difference than globally-optimal solution
		dif_seq = np.where(hist_vec_seq < 10)[0].shape[0]
		dif_clo = np.where(hist_vec_clo < 10)[0].shape[0]
		dif_nav = np.where(hist_vec_nav < 10)[0].shape[0]
		dif_seq_sim = np.where(hist_vec_seq_sim < 10)[0].shape[0]
		print "__<10 Difference________________"
		print "TO: {}%".format((float(dif_seq)/num_inst)*100)
		print "CU: {}%".format((float(dif_clo)/num_inst)*100)
		print "NS: {}%".format((float(dif_nav)/num_inst)*100)
		print "TO+S: {}%".format((float(dif_seq_sim)/num_inst)*100)
		print "\n"

		hist_vec = np.zeros((data_seq.shape[0], 4))

		hist_vec[:,0] = data_seq[:,0] - data_seq[:,1]
		hist_vec[:,1] = data_clo[:,0] - data_clo[:,1]
		hist_vec[:,2] = data_nav[:,0] - data_nav[:,1]
		hist_vec[:,3] = data_seq_sim[:,0] - data_seq_sim[:,1]

		plt.style.use('seaborn-darkgrid')

		plt.hist(	hist_vec, bins=80, normed=True, histtype='step',
					color=['g', 'b', 'k', 'r'],
					label=['TO', 'CU', 'NS', 'TO+S'], stacked=False		)

		plt.xlabel("Difference in solution length")
		plt.ylabel("Probability")
		plt.axis([0, 200, 0, 0.045])
		plt.legend()
		plt.tight_layout()

		plt.savefig("{}/model-solution-length.pdf".format(Utility.getICIPFigureDir()))
		plt.show()

# Entry method/unit testing
if __name__ == '__main__':
	# Utility.drawDatasetSizeAccuracyGraph()
	# Utility.drawAccuracyGraph()

	Utility.drawModelLengthHistogram()

	# print Utility.distanceBetweenPoints((5,0),(0,5))
	# print Utility.distanceBetweenPoints((5,0),(1,5))
	# print Utility.distanceBetweenPoints((5,0),(2,5))
	# print Utility.distanceBetweenPoints((5,0),(3,5))
	# print Utility.distanceBetweenPoints((5,0),(4,5))
	# print Utility.distanceBetweenPoints((5,0),(5,5))
	# print Utility.distanceBetweenPoints((5,0),(6,5))

	# file1_path = "/home/will/catkin_ws/src/uav_id/tflearn/ICIP2018/data/TRAINING_DATA_individual_motion_LARGE.h5"
	# file2_path = "/home/will/catkin_ws/src/uav_id/tflearn/ICIP2018/data/TRAINING_DATA_individual_motion_MS.h5"
	# out_path = "/home/will/catkin_ws/src/uav_id/tflearn/ICIP2018/data/TRAINING_DATA_individual_motion_60k.h5"
	# Utility.combineH5Databases(out_path, file1_path, file2_path)
