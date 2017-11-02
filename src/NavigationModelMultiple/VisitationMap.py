#!/usr/bin/env python

import Utility
import numpy as np
import Constants as const

class MapHandler:
	# Class constructor
	def __init__(self, agent_initial_x, agent_initial_y):
		"""
		Class attributes/properties
		"""

		"""
		Class setup
		"""

		# Reinitialise the map
		self.reset(agent_initial_x, agent_initial_y)

	# Reset the map itself, mark the agent's initial position
	def reset(self, a_x, a_y):
		# Create the map
		self._map = np.zeros((const.MAP_WIDTH, const.MAP_HEIGHT))

		# Fill it with un-visted value
		self._map.fill(const.UNVISITED_VAL)

		# Mark the agent's initial coordinates
		self.setElement(a_x, a_y, const.AGENT_VAL)

	# Update the map to reflect new (given) agent positions
	def update(self, new_x, new_y):
		# Fetch the agent's current position
		curr_x, curr_y = getAgentCoordinates()

		# Make the current agent position a visited location
		self.setElement(curr_x, curr_y, const.VISITED_VAL)

		# Mark the new agent position
		self.setElement(new_x, new_y, const.AGENT_VAL)

	# Set an element at coordinates (x, y) to value
	def setElement(self, x, y, value):
		self._map[y, x] = value

	# Simply prints the map
	def printMap(self):
		print self._map

	# Simply return the map
	def getMap(self):
		return self._map

	# Given a position and the map's boundaries, return a list of possible
	# actions that don't result in the agent going out of bounds
	def possibleActionsForPosition(self, x, y):
		# Get the list of all actions
		actions = const.ACTIONS

		# Check map boundaries in x axis
		if x == 0: actions.remove('L')
		elif x == const.MAP_WIDTH - 1: actions.remove('R')

		# Check map boundaries in y axis
		if y == 0: actions.remove('F')
		elif y == const.MAP_HEIGHT - 1: actions.remove('B')

		return actions

	# Retrieves current agent position from the map
	def getAgentCoordinates(self):
		# Find the current agent position
		index = np.where(self._map == const.AGENT_VAL)

		# Ensure we only found one position
		if index[0].shape[0] > 1 and index[1].shape[0] > 1:
			Utility.die("Found more than one agent location!")

		return index[0][0], index[1][0]

	def findUnvisitedDirection(self, a_x, a_y):
		# Ensure given and knowledge of agent positions match
		temp_x, temp_y = self.getAgentCoordinates()
		assert(a_x == temp_x and a_y == temp_y)

		# Cell search radius for unvisited cells
		radius = 1

		# Determined action to take
		action = None

		# Loop until we find a suitable unvisited direction
		while action is None:
			# Try and find an unvisited location in the current radius
			action = self.determineCellNeighbours(a_x, a_y, radius)

			# Increment the radius
			radius += 1

		return action

	# Given a position x,y return neighbouring values within a given radius
	def determineCellNeighbours(self, x, y, radius):
		# Double the radius
		d_rad = radius * 2

		# Add padding to coordinates
		x += radius
		y += radius

		# Pad a temporary map with ones
		padded_map = np.ones((const.MAP_WIDTH+d_rad, const.MAP_HEIGHT+d_rad))

		# Insert visitation map into border padded map
		padded_map[		radius:const.MAP_WIDTH+radius,
						radius:const.MAP_HEIGHT+radius	] = self._map[:,:]

		# Determine neighbouring cell bounds for radius
		y_low = y - radius
		y_high = y + radius
		x_low = x - radius
		x_high = x + radius

		# Get neighbouring elements within radius (includes x,y-th element)
		sub = padded_map[y_low:y_high+1, x_low:x_high+1]

		# Get indices of elements that are unvisited (0)
		indices = np.where(sub == 0)

		# Action to carry out
		action = None

		# Check whether some 0 elements were found
		if indices[0].size > 0 and indices[1].size > 0:
			# Agent position in subview
			a_x = np.floor((d_rad+1)/2)
			a_y = np.floor((d_rad+1)/2)

			# Random 0 element in subview
			size = indices[1].shape[0]
			rand_element = random.randint(0, size-1)
			z_x = indices[1][rand_element]
			z_y = indices[0][rand_element]

			# Find the best action for the angle between them
			action = Utility.bestActionForAngle((a_x, a_y), (z_x, z_y))

		return action

# Entry method/unit testing
if __name__ == '__main__':
	map_handler = MapHandler(0, 0)
	map_handler.update(1, 1)
