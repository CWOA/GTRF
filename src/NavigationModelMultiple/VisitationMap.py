#!/usr/bin/env python

from Utility import Utility
import numpy as np
import Constants as const
import random

class MapHandler:
	# Class constructor
	def __init__(self):
		"""
		Class attributes/properties
		"""

		print "Initialised MapHandler"

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
		curr_x, curr_y = Utility.getAgentCoordinatesFromMap(self._map)

		# Make the current agent position a visited location
		self.setElement(curr_x, curr_y, const.VISITED_VAL)

		# See whether the agent has already visited the new position
		if self.getElement(new_x, new_y) == const.UNVISITED_VAL: new_location = True
		else: new_location = False 

		# Mark the new agent position
		self.setElement(new_x, new_y, const.AGENT_VAL)

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
	def setElement(self, x, y, value):
		self._map[y, x] = value

# Entry method/unit testing
if __name__ == '__main__':
	Utility.possibleActionsForAngle(1, 1, 0, 0) # Top-left
	Utility.possibleActionsForAngle(1, 1, 0, 2) # Bottom-left
	Utility.possibleActionsForAngle(1, 1, 2, 2) # Bottom-right
	Utility.possibleActionsForAngle(1, 1, 2, 0) # Top-right
	