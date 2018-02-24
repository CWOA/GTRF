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

	# Return the opposing action (e.g. for F, B)
	@staticmethod
	def oppositeAction(action):
		if action == 'F':
			return 'B'
		elif action == 'B':
			return 'F'
		elif action == 'L':
			return 'R'
		elif action == 'R':
			return 'L'

	# Check whether the supplied position is out of map bounds
	@staticmethod
	def checkPositionInBounds(x, y):
		if x < 0 or y < 0 or x >= const.MAP_WIDTH or y >= const.MAP_HEIGHT:
			return False

		return True

	# If all positions in a list are in map boundaries, return true
	@staticmethod
	def checkPositionsListInBounds(pos):
		for p in pos:
			if not Utility.checkPositionInBounds(p[0], p[1]):
				return False

		return True

	# Scale a given value from one range [old_a, old_b] to a new range [new_a, new_b]
	@staticmethod
	def scaleValueFromRangeToRange(val, old_a, old_b, new_a, new_b):
		return (((new_b - new_a) * (val - old_a)) / (old_b - old_a)) + new_a

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

# Entry method/unit testing
if __name__ == '__main__':
	pass

	# file1_path = "/home/will/catkin_ws/src/uav_id/tflearn/ICIP2018/data/TRAINING_DATA_individual_motion_LARGE.h5"
	# file2_path = "/home/will/catkin_ws/src/uav_id/tflearn/ICIP2018/data/TRAINING_DATA_individual_motion_MS.h5"
	# out_path = "/home/will/catkin_ws/src/uav_id/tflearn/ICIP2018/data/TRAINING_DATA_individual_motion_60k.h5"
	# Utility.combineH5Databases(out_path, file1_path, file2_path)
