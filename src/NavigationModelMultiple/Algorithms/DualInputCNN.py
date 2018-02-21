#!/usr/bin/env python

# Core libraries
import sys
sys.path.append('../')

import DNN
import numpy as np
import Constants as const
from Core.VisitationMap import MapHandler

"""
This class contains our dual-input CNN-based solution, for more details, see our
IROS 2018 paper submission entitled "Learning to Search for Distributed Targets
from UAV-like Vision"
"""

class DualInputCNN:
	# Class constructor
	def __init__(	self,
					use_simulator,
					model_path,
					use_loop_detector=True		):
		"""
		Class arguments
		"""

		# Whether loop detection is enabled or not
		self._use_loop_detector = use_loop_detector

		"""
		Class attributes
		"""

		# Initialise the agent loop detection module
		if self._use_loop_detector:
			self._loop_detector = LoopDetector()

		# Deep Neural Network class used for action selection
		self._dnn = DNN.DNNModel(use_simulator)

		"""
		Class initialisation
		"""

		# Signal the DNN to load model weights
		self._dnn.loadModel(model_dir=model_path)

	# Reset function is called each time a new episode is started
	def reset(self):
		# Reset the loop detector
		if self._use_loop_detector:
			self._loop_detector.reset()

		# Counter for the number of times loop detection is triggered
		self._num_loops = 0

		# Indicator of whether the agent is perceived to be sutck in an infinite loop
		# e.g. LRLRLRLRLRLR...
		self._agent_stuck = False

	# Step one iteration for an episode (decide where to move based on input)
	# Input:
	#		(1): Agent-centric visual field of the environment
	#		(2): Occpancy grid map
	# Output:
	#		(1): Action selected by the respective algorithm
	#
	# Given the current agent subview and visit map, use the trained DNN model to predict
	# the best possible action in this circumstance
	def iterate(self, image, occupancy_map):
		# Predict action using DNN
		chosen_action = self.retrieveCNNPrediction(image, occupancy_map)

		# Extract the agent's current position from the occupancy map
		a_x, a_y = Utility.getAgentCoordinatesFromMap(occupancy_map)

		if self._use_loop_detector:
			# Add the suggested action and check history, check if the agent is
			# stuck in a loop, act accordingly
			if not self._agent_stuck and self._loop_detector.addCheckElement(chosen_action, (a_x, a_y)):
				# Indicate that we're stuck
				self._agent_stuck = True
				self._num_loops += 1

			# If the agent is deemed to be stuck
			if self._agent_stuck:
				# Select an appropriate action towards visiting the nearest largest unvisited
				# region of the occupancy map
				chosen_action = MapHandler.findUnvisitedDirection(occupancy_map, a_x, a_y)

				# Get the value of the occupancy map were the action applied
				value = self.elementForAction(occupancy_map, a_x, a_y, chosen_action)

				# If this new position hasn't been visited before, control is handed back to
				# the CNN
				if value == const.UNVISITED_VAL or value == const.MOTION_EMPTY_VAL:
					# Delete elements in the loop detector
					self._loop_detector.reset()

					# Indicate that the agent is no longer stuck
					self._agent_stuck = False

		return chosen_action

	# Use the trained dual-input CNN in order to predict a suitable action based on the
	# given inputs
	def retrieveCNNPrediction(self, image, occupancy_map):
		# Predict using DNN, returns probabilty score list for each class
		probability_vec = self._dnn.testModelSingle(image, occupancy_map)

		# Find index of max value
		max_idx = np.argmax(probability_vec)

		# Create a new probability vector with the max index = 1
		if const.USE_EXT_ACTIONS:
			choice_vec = np.zeros(len(const.EXT_ACTIONS))
		else:
			choice_vec = np.zeros(len(const.ACTIONS))

		choice_vec[max_idx] = 1

		# Convert to action
		action = Utility.classVectorToAction(choice_vec)

		return action

	# Return the occupancy map element for a prospective coordinate based on a selected
	# action
	def elementForAction(self, occupancy_map, a_x, a_y, action):
		# New coordinates
		new_x = a_x
		new_y = a_y

		# Find coordinates based on chosen action
		if action == 'F': 	new_y -= const.MOVE_DIST
		elif action == 'B': new_y += const.MOVE_DIST
		elif action == 'L': new_x -= const.MOVE_DIST
		elif action == 'R': new_x += const.MOVE_DIST
		else: Utility.die("Action: {} not recognised!".format(action), __file__)

		# Check the position is in the map's boundaries
		if not Utility.checkPositionInBounds(new_x, new_y):
			Utility.die("Action: {} for new position ({},{}) is out of bounds".\
							format(action, new_x, new_y), __file__)

		# Return the map value at the prospective position
		if const.OCCUPANCY_MAP_MODE == const.VISITATION_MODE:
			return occupancy_map[new_y, new_x]
		elif const.OCCUPANCY_MAP_MODE == const.MOTION_MODE:
			return occupancy_map[new_y, new_x, 1]
		else: Utility.die("Occupancy map mode not recognised in elementForAction()", __file__)

	"""
	Getters
	"""
	def getNumLoops(self):
		return self._num_loops

	"""
	Setters
	"""

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
