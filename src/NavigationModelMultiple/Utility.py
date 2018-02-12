#!/usr/bin/env python

import os
import sys
import math
import random
import numpy as np
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
	def getBestModelDir():
		filename = "{}_BEST.tflearn".format(const.MODEL_NAME)
		return os.path.join(const.BASE_DIR, const.MODELS_DIR, filename)
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
	def getICIPBestModelDir():
		return os.path.join(const.BASE_DIR, const.ICIP_MODELS_DIR, "best/")
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
		vec = np.zeros(len(const.ACTIONS))

		if action == 'F': vec[0] = 1
		elif action == 'B': vec[1] = 1
		elif action == 'L': vec[2] = 1
		elif action == 'R': vec[3] = 1
		else: Utility.die("Action not recognised.", __file__)

		return vec

	# The opposite of the above function
	@staticmethod
	def classVectorToAction(class_vec):
		action = ''

		if class_vec[0]: action = 'F'
		elif class_vec[1]: action = 'B'
		elif class_vec[2]: action = 'L'
		elif class_vec[3]: action = 'R'
		else: Utility.die("Action not recognised.", __file__)

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
		index = np.where(occupancy_map == const.AGENT_VAL)

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

	"""
	Graph drawing utility methods for evaluating algorithm performance
	"""

	@staticmethod
	def drawGenerationTimeGraph(t_s, t_c, s_t, e_t):
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
		# Construct size of targets vector
		T = np.arange(s_t, e_t)

		seq_avg = np.average(m_s, axis=1)
		clo_avg = np.average(m_c, axis=1)

		# print seq_avg
		# print clo_avg

		hist_vec = (m_c - m_s).flatten()

		# print hist_vec

		N, bins, patches = plt.hist(hist_vec, bins=13, normed=True)

		# Plot vectors
		# plt.plot(T, seq_avg, label="Sequence")
		# plt.plot(T, clo_avg, label="Closest")

		# Graph parameters
		plt.xlabel('Difference in solution length')
		# plt.ylabel('moves')
		plt.legend(loc="center right")
		plt.savefig("{}/solution-generation-hist.pdf".format(Utility.getICIPFigureDir()))
		plt.show()

	@staticmethod
	def drawGenerationGraphs(m_s, m_c, t_s, t_c, num_targets):
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

		plt.savefig("{}/solution-generation.pdf".format(Utility.getICIPFigureDir()))
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

		N, bins, patches = plt.hist(hist_vec, bins=80, normed=True, histtype='step',
									color=['g', 'b', 'k', 'r'],
									label=['TO', 'CU', 'NS', 'TO+S'], stacked=False)

		plt.xlabel("Difference in solution length")
		plt.ylabel("Probability")
		plt.axis([0, 200, 0, 0.045])
		plt.legend()
		plt.tight_layout()

		plt.savefig("{}/model-solution-length.pdf".format(Utility.getICIPFigureDir()))
		plt.show()

# Entry method/unit testing
if __name__ == '__main__':
	# Utility.drawModelLengthHistogram()

	# print Utility.distanceBetweenPoints((5,0),(0,5))
	# print Utility.distanceBetweenPoints((5,0),(1,5))
	# print Utility.distanceBetweenPoints((5,0),(2,5))
	# print Utility.distanceBetweenPoints((5,0),(3,5))
	# print Utility.distanceBetweenPoints((5,0),(4,5))
	# print Utility.distanceBetweenPoints((5,0),(5,5))
	# print Utility.distanceBetweenPoints((5,0),(6,5))
