#!/usr/bin/env python

# Core functions
import sys
sys.path.append('../')
import numpy as np
import Constants as const
import random

# My libraries/classes
from Utilities.Utility import Utility

"""
This class manages the occupancy/visitation map map
"""

class MapHandler:
	# Class constructor
	def __init__(	self,
					map_mode=const.OCCUPANCY_MAP_MODE,
					mark_visitation=const.MARK_PAST_VISITATION	):
		"""
		Class attributes/properties
		"""

		# Whether we should mark past target visitation
		self._mark_visitation = mark_visitation

		# Whether this map should operate in visitation (static) or gaussian (moving) mode 
		self._map_mode = map_mode

	"""
	Class methods
	"""

	# Reset the map itself, mark the agent's initial position
	def reset(self, a_x, a_y):
		# If the map is in visitation mode
		if self._map_mode == const.VISITATION_MODE:
			# Create the map
			self._map = np.zeros((const.MAP_WIDTH, const.MAP_HEIGHT))

			# Fill it with un-visted value
			self._map.fill(const.UNVISITED_VAL)

			# Mark the agent's initial coordinates
			self.setElement(a_x, a_y, const.AGENT_VAL)

			# If we should mark coordinates where a target was visited
			if self._mark_visitation:
				# Store where targets were visited
				self._visit_locations = []

		# If the map is in motion mode:
		# 1st 10x10 grid: Where targets have been visited
		# 2nd 10x10 grid: Where has agent been previously
		elif self._map_mode == const.MOTION_MODE:
			# Create the map
			self._map = np.zeros((const.MAP_WIDTH, const.MAP_HEIGHT, 2))

			# Set all values to empty
			self._map.fill(const.MOTION_EMPTY_VAL)

			# Mark the agent's initial coordinates
			self.setElement(a_x, a_y, const.MOTION_HIGH_VALUE, dim=1)

			# Past target locations (dict of target ID: x,y coordinates, steps since visit)
			self._visit_locations = {}

	# Update the map to reflect new (given) agent positions
	def iterate(self, new_x, new_y, target_match, target_id):
		# Fetch the agent's current position
		curr_x, curr_y = Utility.getAgentCoordinatesFromMap(self._map)

		# Whether the agent is now at a new (unvisited) location
		new_location = False

		# If the map is visitation mode
		if self._map_mode == const.VISITATION_MODE:
			# Make the current agent position a visited location
			self.setElement(curr_x, curr_y, const.VISITED_VAL)

			# See whether the agent has already visited the new position
			if self.getElement(new_x, new_y) == const.UNVISITED_VAL: new_location = True

			# If we should mark coordinates where a target was visited
			if self._mark_visitation:
				# If the agent visited a new target this iteration
				if target_match:
					# Add this location
					self._visit_locations.append((new_x, new_y))

				# Render all visit locations
				for location in self._visit_locations:
					self.setElement(location[0], location[1], const.TARGET_VISITED_VAL)

			# Mark the new agent position
			self.setElement(new_x, new_y, const.AGENT_VAL)
		# Map is gaussian probabiltistic mode
		elif self._map_mode == const.MOTION_MODE:
			# Unmark the entire target visitation dimension
			self._map[:,:,0] = const.MOTION_EMPTY_VAL

			# Increment the step counter for each visited target
			for key in self._visit_locations:
				# Degrade the time since visitation
				self._visit_locations[key][2] -= 1

				# Cap it at 0
				if self._visit_locations[key][2] < 0:
					self._visit_locations[key][2] = 0

			# If this new position visits a target
			if target_match:
				self._visit_locations[target_id] = [new_x, new_y, const.MOTION_HIGH_VALUE]

			# Render target visits
			for _, value in self._visit_locations.iteritems():
				self.setElement(value[0], value[1], value[2])

			# Degrade every non-zero value in the agent visitation dimension
			self._map[np.where(self._map[:,:,1] > 0)] -= 1

			# Mark the agent's new position
			self.setElement(new_x, new_y, const.MOTION_HIGH_VALUE, dim=1)
		else:
			Utility.die("Occupancy map mode not recognised in iterate()", __file__)

		return new_location

	"""
	Agent un-stucking methods

	These functions need to be accessed statically sometimes
	"""

	# Call static function with this instance's map
	def findUnvisitedDirectionNonStatic(self, a_x, a_y):
		return MapHandler.findUnvisitedDirection(self.getMap(), a_x, a_y)

	# Given that the agent is deemed to be an infinite loop
	@staticmethod
	def findUnvisitedDirection(o_map, a_x, a_y):
		# Cell search radius for unvisited cells
		radius = 1

		# Determined action to take
		action = None

		# Loop until we find a suitable unvisited direction
		while action is None:
			# Try and find an unvisited location in the current radius
			action = MapHandler.determineCellNeighbours(o_map, a_x, a_y, radius)

			# Increment the radius
			radius += 1

		return action

	# Given a position x,y return neighbouring values within a given radius
	@staticmethod
	def determineCellNeighbours(o_map, x, y, radius):
		# Double the radius
		d_rad = radius * 2

		# Add padding to coordinates
		x += radius
		y += radius

		# Pad a temporary map with ones
		padded_map = np.ones((const.MAP_WIDTH+d_rad, const.MAP_HEIGHT+d_rad))

		if const.OCCUPANCY_MAP_MODE == const.VISITATION_MODE:
			# Insert visitation map into border padded map
			padded_map[		radius:const.MAP_WIDTH+radius,
							radius:const.MAP_HEIGHT+radius	] = o_map[:,:]
		elif const.OCCUPANCY_MAP_MODE == const.MOTION_MODE:
			# Insert visitation map into border padded map
			padded_map[		radius:const.MAP_WIDTH+radius,
							radius:const.MAP_HEIGHT+radius	] = o_map[:,:,1]
		else:
			Utility.die("Occupancy map mode not recognised in determineCellNeighbours()", __file__)

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
			Utility.die("All cells have been visited..", __file__)

		# Action to carry out
		action = None

		# Check whether some 0 elements were found
		if indices[0].size > 0 and indices[1].size > 0:
			# Agent position in subview
			a_x = np.floor((d_rad+1)/2)
			a_y = np.floor((d_rad+1)/2)

			# Find the best action for the angle between them
			action = MapHandler.bestActionForCoordinates(a_x, a_y, indices[1], indices[0])

		return action

	# Construct the action vote table, find the most voted for action
	@staticmethod
	def bestActionForCoordinates(a_x, a_y, indices_x, indices_y):
		# Construct vote table of possible actions
		vote_table = MapHandler.constructActionVoteTable(a_x, a_y, indices_x, indices_y)

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

	# Actually construct the action vote table
	@staticmethod
	def constructActionVoteTable(a_x, a_y, indices_x, indices_y):
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
			vote_table = MapHandler.incrementActionCounts(vote_table, possible_actions)

		return vote_table

	# Given the current action vote table, increment elements that are present in 
	# the provided action vector
	@staticmethod
	def incrementActionCounts(table, action_vec):
		for action in action_vec:
			if action == 'F': table[0] += 1
			elif action == 'B':  table[1] += 1
			elif action == 'L': table[2] += 1
			elif action == 'R': table[3] += 1
			else: Utility.die("Action not recognised", __file__)

		return table

	"""
	Getters
	"""

	def getElement(self, x, y):
		return self._map[y, x]
	def printMap(self):
		if self._map_mode == const.VISITATION_MODE:
			print self._map
		elif self._map_mode == const.MOTION_MODE:
			print self._map[:,:,0], self._map[:,:,1]
		else:
			Utility.die("Occupancy map mode not recognised in setElement()", __file__)
	def getMap(self):
		return self._map

	"""
	Setters
	"""

	# Set an element at coordinates (x, y) to value
	def setElement(self, x, y, value, dim=0):
		if self._map_mode == const.VISITATION_MODE:
			self._map[y, x] = value
		elif self._map_mode == const.MOTION_MODE:
			self._map[y, x, dim] = value
		else:
			Utility.die("Occupancy map mode not recognised in setElement()", __file__)

# Entry method/unit testing
if __name__ == '__main__':
	Utility.possibleActionsForAngle(1, 1, 0, 0) # Top-left
	Utility.possibleActionsForAngle(1, 1, 0, 2) # Bottom-left
	Utility.possibleActionsForAngle(1, 1, 2, 2) # Bottom-right
	Utility.possibleActionsForAngle(1, 1, 2, 0) # Top-right
	