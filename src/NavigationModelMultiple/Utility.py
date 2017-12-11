#!/usr/bin/env python

import os
import sys
import math
import random
import numpy as np
import Constants as const
from collections import deque
import matplotlib.pyplot as plt

# Utility class for static methods
class Utility:
	### Directory functions
	@staticmethod
	def getPickleDataDir():
		return os.path.join(const.BASE_DIR, const.DATA_DIR_PICKLE)
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
	@staticmethod
	def getICIPDataDir():
		return os.path.join(const.BASE_DIR, const.ICIP_DATA_DIR)

	# Compute the shortest path action sequence from a -> b
	@staticmethod
	def actionSequenceBetweenCoordinates(a_x, a_y, b_x, b_y):
		actions = []

		# Loop until we're at the ending position
		while (a_x, a_y) != (b_x, b_y):
			# Find possible actions for the current-end relative vector
			possible_actions = Utility.possibleActionsForAngle(a_x, a_y, b_x, b_y)

			# Randomly select an action
			rand_idx = random.randint(0, len(possible_actions)-1)
			choice = possible_actions[rand_idx]

			# Perform the chosen action
			if choice == 'F': 	a_y -= const.MOVE_DIST
			elif choice == 'B': a_y += const.MOVE_DIST
			elif choice == 'L': a_x -= const.MOVE_DIST
			elif choice == 'R': a_x += const.MOVE_DIST
			else: Utility.die("Action: {} not recognised!".format(choice))

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
		else: Utility.die("Action not recognised.")

		return vec

	# The opposite of the above function
	@staticmethod
	def classVectorToAction(class_vec):
		action = ''

		if class_vec[0]: action = 'F'
		elif class_vec[1]: action = 'B'
		elif class_vec[2]: action = 'L'
		elif class_vec[3]: action = 'R'
		else: Utility.die("Action not recognised.")

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
		else: Utility.die("Angle is not in [0,360] degrees")

		# Make sure the assigned action is valid
		assert(action in const.ACTIONS)

		return action

	# Very similar to "bestActionForAngle" except for the case when an angle is 45 degrees
	# it returns both F, R in a char vector
	@staticmethod
	def possibleActionsForAngle(a_x, a_y, b_x, b_y):
		# Compute angle between given points
		angle = Utility.angleBetweenPoints(a_x, a_y, b_x, b_y)

		# Pi!
		p = math.pi

		# If the angle is exactly diagonal (in 45 degree increments)
		# top left 
		if angle == p/4: return ['F', 'L']
		elif angle == 3*p/4: return ['L', 'B']
		elif angle == -3*p/4: return ['B', 'R']
		elif angle == -p/4: return ['R', 'F']
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

	@staticmethod
	def die(message):
		print "{}\nExiting..".format(message)
		sys.exit(0)

	"""
	Graph drawing utility methods
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

# Class designed to help with detecting whether the agent is stuck in an infinite loop
class LoopDetector:
	# Class constructor
	def __init__(	self, 
					use_action_strings=const.USE_ACTION_STRING, 
					max_queue_size=const.MAX_QUEUE_SIZE,
					max_visit_occurences=const.MAX_VISIT_OCCURENCES		):
		"""
		Class attributes
		"""

		# Whether to use string based loop detection method or coordinate system
		self._use_action_strings = use_action_strings

		# Maximum length of queue
		self._max_queue_size = max_queue_size

		# Maximum number of times one coordinate is allowed to occur before a loop is
		# declared
		self._max_visits = max_visit_occurences

		print "Initialised LoopDetector"

	# Reset so we can start a new instance
	def reset(self):
		# Queue to store past actions
		self._list = deque()

	# Add a position/coordinate and check the queue for a loop
	def addCheckElement(self, action, coordinate):
		# If we're using string-based or coordinate system
		if self._use_action_strings: self.addElementToQueue(action)
		else: self.addElementToQueue(coordinate)

		return self.checkForLoop()

	# Add an action to the queue
	def addElementToQueue(self, action):
		# Add the action
		self._list.append(action)

		# Check the length of the queue
		if len(self._list) == self._max_queue_size + 1:
			# We need to pop an older entry
			self._list.popleft()

	# Given the current action queue, detect whether a loop has occurred
	def checkForLoop(self):
		# If we're using string-based or coordinate system
		if self._use_action_strings:
			if self.checkActionSequenceSubstring("FBF"): return True
			if self.checkActionSequenceSubstring("BFB"): return True
			if self.checkActionSequenceSubstring("LRL"): return True
			if self.checkActionSequenceSubstring("RLR"): return True
			if self.checkActionSequenceRotationReverse("RBLF"): return True
			if self.checkActionSequenceRotationReverse("RRBLLF"): return True
			if self.checkActionSequenceRotationReverse("RBBLFF"): return True
			if self.checkActionSequenceRotationReverse("RRFFBBLL"): return True
		# Use alternative coordinate system
		else:
			if self.checkCoordinateQueue(): return True

		return False

	"""
	COORDINATE-BASED functions
	"""

	# Check the coordinate queue to see whether locations have occurred multiple times
	def checkCoordinateQueue(self):
		for item in self._list:
			if self._list.count(item) >= self._max_visits:
				return True

		return False

	"""
	STRING-BASED functions
	"""

	# Check for a substring in the actual sequence
	def checkActionSequenceSubstring(self, sequence):
		# Convert list of characters to an ordered string
		actual = ''.join(self._list)

		# Supplied substring is present in actual sequence string
		if sequence in actual:
			return True

		return False

	# Check actual sequence for given sequence with all possible rotations (shifts)
	# e.g. RBLF, FRBL, LFRB, ...
	def checkActionSequenceRotation(self, sequence):
		for i in range(len(sequence)):
			rotated = Utility.rotateList(sequence, i)
			if self.checkActionSequenceSubstring(''.join(rotated)):
				return True

		return False

	# *** Also check the reverse of the given sequence
	def checkActionSequenceRotationReverse(self, sequence):
		# Convert to list of characters
		sequence_char_list = list(sequence)

		# Check forwards
		if self.checkActionSequenceRotation(sequence_char_list): return True

		# Reverse the list
		sequence_char_list.reverse()

		# Check the reverse
		if self.checkActionSequenceRotation(sequence_char_list): return True

		return False

# Entry method/unit testing
if __name__ == '__main__':
	pass
