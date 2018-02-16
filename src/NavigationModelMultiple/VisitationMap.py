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

			# If we should mark coordinates where a target was visited
			if const.MARK_PAST_VISITATION:
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
			if const.MARK_PAST_VISITATION:
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
	