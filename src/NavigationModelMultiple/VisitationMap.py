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
		curr_x, curr_y = self.getAgentCoordinates()

		# Make the current agent position a visited location
		self.setElement(curr_x, curr_y, const.VISITED_VAL)

		# See whether the agent has already visited the new position
		if self.getElement(new_x, new_y) == const.UNVISITED_VAL: new_location = True
		else: new_location = False 

		# Mark the new agent position
		self.setElement(new_x, new_y, const.AGENT_VAL)

		return new_location

	def getElement(self, x, y):
		return self._map[y, x]

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
		actions = list(const.ACTIONS)

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

		return index[1][0], index[0][0]

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

		# If that element is only found once
		if len(other_idx) == 1:
			action = const.ACTIONS[max_idx]
		# Randomly chose an index and action
		else:
			rand_idx = random.randint(0, other_idx.size - 1)
			action = const.ACTIONS[rand_idx]

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

	# Check whether the supplied position is out of bounds
	def checkPositionInBounds(self, x, y):
		if x < 0 or y < 0 or x >= const.MAP_WIDTH or y >= const.MAP_HEIGHT:
			return False

		return True

# Entry method/unit testing
if __name__ == '__main__':
	# map_handler = MapHandler(0, 0)
	# map_handler.update(1, 1)

	Utility.possibleActionsForAngle(1, 1, 0, 0) # Top-left
	Utility.possibleActionsForAngle(1, 1, 0, 2) # Bottom-left
	Utility.possibleActionsForAngle(1, 1, 2, 2) # Bottom-right
	Utility.possibleActionsForAngle(1, 1, 2, 0) # Top-right