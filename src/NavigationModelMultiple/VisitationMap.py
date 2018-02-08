#!/usr/bin/env python

from Utility import Utility
import numpy as np
import Constants as const
import random

"""
This class manages the occupancy map
"""

class MapHandler:
	# Class constructor
	def __init__(	self,
					map_mode=const.OCCUPANCY_MAP_MODE	):
		"""
		Class attributes/properties
		"""

		# Whether this map should operate in visitation (static) or gaussian (moving) mode 
		self._map_mode = map_mode

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
		# If the map is in Gaussian mode
		elif self._map_mode == const.GAUSSIAN_MODE:
			# Create the map with an extra dimension
			# The first contains sigma values
			# The second contains the number of targets at that point
			self._map = np.zeros((const.MAP_WIDTH, const.MAP_HEIGHT, 2))

			# Fill it with un-visited value
			self._map.fill(const.UNVISITED_VAL)

			# Mark the agent's initial coordinates
			self.setElement(a_x, a_y, const.AGENT_VAL)

			# Find the center coordinates of the map
			c_x = int(round(const.MAP_WIDTH/2))
			c_y = int(round(const.MAP_HEIGHT/2))

			# Initial sigma value
			sigma_init = c_x

			# Mark all target's distribution center
			self.setElement(c_x, c_y, sigma_init, dim=0)

			# Mark the number of targets at the center
			self.setElement(c_x, c_y, const.NUM_TARGETS, dim=1)

	# Update the map to reflect new (given) agent positions
	def iterate(self, new_x, new_y):
		# Fetch the agent's current position
		curr_x, curr_y = Utility.getAgentCoordinatesFromMap(self._map)

		# If the map is visitation mode
		if self._map_mode == const.VISITATION_MODE:
			# Make the current agent position a visited location
			self.setElement(curr_x, curr_y, const.VISITED_VAL)

			# See whether the agent has already visited the new position
			if self.getElement(new_x, new_y) == const.UNVISITED_VAL: new_location = True
			else: new_location = False

			# Mark the new agent position
			self.setElement(new_x, new_y, const.AGENT_VAL)
		# Map is gaussian probabiltistic mode
		elif self._map_mode == const.GAUSSIAN_MODE:
			# Unmark the agent's current position
			self.setElement(curr_x, curr_y, const.UNVISITED_VAL)

			# Mark the agent's new position
			self.setElement(new_x, new_y, const.AGENT_VAL)
		else:
			Utility.die("Occupancy map mode not recognised", __file__)


		return new_location

	# Given a position and the map's boundaries, return a list of possible
	# actions that don't result in the agent going out of bounds
	def possibleActionsForPosition(self, x, y):
		# Get the list of all actions
		actions = list(const.ACTIONS)

		# Check map boundaries in x axis
		if x == 0: actions.remove('L')
		elif x == const.MAP_WIDTH - 1: actions.remove('R')

		# Check map boundaries in y axis
		if y == 0: actions.remove('F')
		elif y == const.MAP_HEIGHT - 1: actions.remove('B')

		return actions

	"""
	Getters
	"""

	def getElement(self, x, y):
		return self._map[y, x]
	def printMap(self):
		print self._map
	def getMap(self):
		return self._map

	"""
	Setters
	"""

	# Set an element at coordinates (x, y) to value
	def setElement(self, x, y, value, dim=0):
		if self._map_mode == const.VISITATION_MODE:
			self._map[y, x] = value
		# Gaussian mode has an extra dimension
		elif self._map_mode == const.GAUSSIAN_MODE:
			self._map[y, x, dim] = value

# Entry method/unit testing
if __name__ == '__main__':
	Utility.possibleActionsForAngle(1, 1, 0, 0) # Top-left
	Utility.possibleActionsForAngle(1, 1, 0, 2) # Bottom-left
	Utility.possibleActionsForAngle(1, 1, 2, 2) # Bottom-right
	Utility.possibleActionsForAngle(1, 1, 2, 0) # Top-right
	