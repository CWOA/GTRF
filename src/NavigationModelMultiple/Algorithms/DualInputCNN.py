#!/usr/bin/env python

import DNN
from Utility import *
import numpy as np
import Constants as const

"""
This class contains our dual-input CNN-based solution, for more details, see our
IROS 2018 paper submission entitled "Learning to Search for Distributed Targets
from UAV-like Vision"
"""

class DualInputCNN:
	# Class constructor
	def __init__(	self,
					use_simulator	):
		"""
		Class attributes
		"""

		# Initialise the agent loop detection module
		self._loop_detector = LoopDetector()

		# Deep Neural Network class used for action selection
		self._dnn = DNN.DNNModel(use_simulator)

		"""
		Class initialisation
		"""

		# Signal the DNN to load model weights
		self._dnn.loadModel()

	# Reset function is called each time a new episode is started
	def reset(self):
		# Reset the loop detector
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

		# Add the suggested action and check history, check if the agent is
		# stuck in a loop, act accordingly
		if not self._agent_stuck and self._loop_detector.addCheckElement(chosen_action, (a_x, a_y)):
			# Indicate that we're stuck
			self._agent_stuck = True
			self._num_loops += 1

			print "Agent stuck"

		# If the agent is deemed to be stuck
		if self._agent_stuck:
			# Select an appropriate action towards visiting the nearest largest unvisited
			# region of the occupancy map
			chosen_action = self.findUnvisitedDirection(occupancy_map, a_x, a_y)

			# Get the value of the occupancy map were the action applied
			value = self.elementForAction(occupancy_map, a_x, a_y, chosen_action)

			# If this new position hasn't been visited before, control is handed back to
			# the CNN
			if value == const.UNVISITED_VAL:
				# Delete elements in the loop detector
				self._loop_detector.reset()

				# Indicate that the agent is no longer stuck
				self._agent_stuck = False

				print "UNSTUCK"

		return chosen_action

	# Use the trained dual-input CNN in order to predict a suitable action based on the
	# given inputs
	def retrieveCNNPrediction(self, image, occupancy_map):
		# Predict using DNN, returns probabilty score list for each class
		probability_vec = self._dnn.testModelSingle(image, occupancy_map)

		# Find index of max value
		max_idx = np.argmax(probability_vec)

		# Create a new probability vector with the max index = 1
		choice_vec = np.zeros(len(const.ACTIONS))
		choice_vec[max_idx] = 1

		# Convert to action
		return Utility.classVectorToAction(choice_vec)

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
		else: Utility.die("Action: {} not recognised!".format(action))

		# Check the position is in the map's boundaries
		if not Utility.checkPositionInBounds(new_x, new_y):
			Utility.die("Action: {} for new position ({},{}) is out of bounds".\
							format(action, new_x, new_y))

		# Return the map value at the prospective position
		return occupancy_map[new_y, new_x]

	# Given that the agent is deemed to be an infinite loop
	def findUnvisitedDirection(self, o_map, a_x, a_y):
		# Cell search radius for unvisited cells
		radius = 1

		# Determined action to take
		action = None

		# Loop until we find a suitable unvisited direction
		while action is None:
			# Try and find an unvisited location in the current radius
			action = self.determineCellNeighbours(o_map, a_x, a_y, radius)

			# Increment the radius
			radius += 1

		return action

	# Given a position x,y return neighbouring values within a given radius
	def determineCellNeighbours(self, o_map, x, y, radius):
		# Double the radius
		d_rad = radius * 2

		# Add padding to coordinates
		x += radius
		y += radius

		# Pad a temporary map with ones
		padded_map = np.ones((const.MAP_WIDTH+d_rad, const.MAP_HEIGHT+d_rad))

		# Insert visitation map into border padded map
		padded_map[		radius:const.MAP_WIDTH+radius,
						radius:const.MAP_HEIGHT+radius	] = o_map[:,:]

		# Determine neighbouring cell bounds for radius
		y_low = y - radius
		y_high = y + radius
		x_low = x - radius
		x_high = x + radius

		# Get neighbouring elements within radius (includes x,y-th element)
		sub = padded_map[y_low:y_high+1, x_low:x_high+1]

		try:
			# Get indices of elements that are unvisited (0)
			indices = np.where(sub == 0)
		except Exception as e:
			Utility.die("All cells have been visited..")

		# Action to carry out
		action = None

		# Check whether some 0 elements were found
		if indices[0].size > 0 and indices[1].size > 0:
			# Agent position in subview
			a_x = np.floor((d_rad+1)/2)
			a_y = np.floor((d_rad+1)/2)

			# Find the best action for the angle between them
			action = self.bestActionForCoordinates(a_x, a_y, indices[1], indices[0])

		return action

	# Construct the action vote table, find the most voted for action
	def bestActionForCoordinates(self, a_x, a_y, indices_x, indices_y):
		# Construct vote table of possible actions
		vote_table = self.constructActionVoteTable(a_x, a_y, indices_x, indices_y)

		# Find first occurence of most voted for action
		max_idx = np.argmax(vote_table)

		# What's the vote count for the most voted for
		element = vote_table[max_idx]

		# Is that element anywhere else in the vote table?
		other_idx = np.where(vote_table == element)

		# Find the size
		other_idx_size = len(other_idx[0])

		# If that element is only found once
		if other_idx_size == 1:
			action = const.ACTIONS[max_idx]
		# Randomly chose an index and action
		else:
			rand_idx = random.randint(0, other_idx_size - 1)
			action = const.ACTIONS[other_idx[0][rand_idx]]

		return action

	def constructActionVoteTable(self, a_x, a_y, indices_x, indices_y):
		# Check the coordinate vectors are equal in size
		assert(indices_x.size == indices_y.size)

		# Initialise vote table to the number of all actions
		vote_table = np.zeros(len(const.ACTIONS))

		# Iterate over each coordinate
		for i in range(indices_x.size):
			# Extract current coordinates
			b_x = indices_x[i]
			b_y = indices_y[i]

			# Get a vector of possible actions for this position
			possible_actions = Utility.possibleActionsForAngle(a_x, a_y, b_x, b_y)

			# Increment vote table with suitable actions returned for these coordinates
			vote_table = self.incrementActionCounts(vote_table, possible_actions)

		return vote_table

	# Given the current action vote table, increment elements that are present in 
	# the provided action vector
	def incrementActionCounts(self, table, action_vec):
		for action in action_vec:
			if action == 'F': table[0] += 1
			elif action == 'B':  table[1] += 1
			elif action == 'L': table[2] += 1
			elif action == 'R': table[3] += 1
			else: Utility.die("Action not recognised")

		return table

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
