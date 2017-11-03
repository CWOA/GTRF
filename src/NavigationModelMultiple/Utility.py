#!/usr/bin/env python

import os
import sys
import math
import Constants as const
from collections import deque

# Utility class for static methods
class Utility:
	### Directory functions
	@staticmethod
	def getDataDir():
		return os.path.join(const.BASE_DIR, const.DATA_DIR)
	@staticmethod
	def getTensorboardDir():
		return os.path.join(const.BASE_DIR, const.TENSORBOARD_DIR)
	@staticmethod
	def getModelDir():
		filename = "{}.tflearn".format(const.MODEL_NAME)
		return os.path.join(const.BASE_DIR, const.MODELS_DIR, filename)

	# Converts from a single action to a class vector required by the dnn model
	# e.g. 'F' -> [1,0,0,0]
	@staticmethod
	def actionToClassVector(action):
		vec = np.zeros(len(self._actions))

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
		# Get relative position
		rel_x = a_x - b_x
		rel_y = a_y - b_y

		# Compute angle
		angle = math.atan2(rel_x, rel_y)

		# print "Angle = {} for point ({},{})".format(math.degrees(angle), rel_x, rel_y)

		if angle < math.pi/4 and angle > -math.pi/4: action = 'F'
		elif angle >= math.pi/4 and angle < 3*math.pi/4: action = 'L'
		elif angle <= math.pi/4 and angle > -3*math.pi/4: action = 'R'
		elif angle >= 3*math.pi/4 or angle <= -3*math.pi/4: action = 'B'

		# Make sure the assigned action is valid
		assert(action in const.ACTIONS)

		return action

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

# Class designed to help with detecting whether the agent is stuck in an infinite loop
class LoopDetector:
	# Class constructor
	def __init__(self, max_queue_size=8):
		"""
		Class attributes
		"""

		# Maximum length of queue
		self._max_queue_size = max_queue_size

		print "Initialised LoopDetector"

	# Reset so we can start a new instance
	def reset(self):
		# Queue to store past actions
		self._actions = deque()

	# Add an action and check the queue
	def addCheckAction(self, action):
		self.addActionToQueue(action)
		return self.checkForLoop()

	# Add an action to the queue
	def addActionToQueue(self, action):
		# Add the action
		self._actions.append(action)

		# Check the length of the queue
		if len(self._actions) == self._max_queue_size + 1:
			# We need to pop an older entry
			self._actions.popleft()

	# Check whether the supplied sequence and actual sequence are exactly equal
	def checkActionSequence(self, sequence):
		equal = True

		if len(sequence) == len(self._actions):
			for i in range(len(sequence)):
				if sequence[i] != self._actions[i]:
					equal = False
					break
		else:
			return False

		return equal

	# Check for a substring in the actual sequence
	def checkActionSequenceSubstring(self, sequence):
		# Convert list of characters to an ordered string
		actual = ''.join(self._actions)

		# Supplied substring is present in actual sequence string
		if sequence in actual:
			return True

		return False

	# Check actual sequence for given sequence with all possible rotations (shifts)
	# e.g. RBLF, FRBL, LFRB, ...
	def checkActionSequenceRotation(self, sequence):
		for i in range(len(sequence)):
			rotated = Utility.rotateList(sequence, i)
			if self.checkActionSequence(rotated):
				return True

		return False

	# *** Also check the reverse of the given sequence
	def checkActionSequenceRotationReverse(self, sequence):
		# Convert to list of characters
		sequence = list(sequence)

		# Check forwards
		if self.checkActionSequenceRotation(sequence): return True
		
		# Reverse the list
		sequence.reverse()

		# Check the reverse
		if self.checkActionSequenceRotation(sequence): return True

		return False

	# Given the current action queue, detect whether a loop has occurred
	def checkForLoop(self):
		if self.checkActionSequenceSubstring("FBF"): return True
		if self.checkActionSequenceSubstring("BFB"): return True
		if self.checkActionSequenceSubstring("LRL"): return True
		if self.checkActionSequenceSubstring("RLR"): return True
		if self.checkActionSequenceRotationReverse("RBLF"): return True
		if self.checkActionSequenceRotationReverse("RRBLLF"): return True
		if self.checkActionSequenceRotationReverse("RBBLFF"): return True
		if self.checkActionSequenceRotationReverse("RRFFBBLL"): return True

		return False
