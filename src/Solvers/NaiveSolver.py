#!/usr/bin/env python

# Core libraries
import sys
sys.path.append('../')
import time
import copy
import math
import random
import networkx as nx
import matplotlib.pyplot as plt

# My libraries/classes
import Constants as const
from Utilities.Utility import Utility
from Utilities.DiscoveryRate import DiscoveryRate
from Core.VisitationMap import MapHandler

"""
This class is for "attempting" to solve an episode algorithmically to provide a baseline
details on how it operates:
TBC
"""

class NaiveSolver:
	# Class constructor
	def __init__(self):
		"""
		Class attributes
		"""

		# Occupancy/visitation map
		self._v_map = MapHandler()

		# Target discovery rate handler
		self._dr = DiscoveryRate()

	"""
	Mandatory class methods
	"""

	def reset(self, agent, targets):
		# Objects for the agent and all targets
		self._agent = copy.deepcopy(agent)
		self._targets = copy.deepcopy(targets)

		# Number of target objects
		self._num_targets = len(targets)

		# Reset the visitation map
		self._v_map.reset(*self._agent.getPos())

		self._num_visited = 0

		# Reset the discovery rate class
		self._dr.reset()

	def solve(self):
		self._actions, mu_DT = self.mainLoop()
		return len(self._actions), mu_DT

	def nextAction(self):
		return self._actions.pop(0)

	"""
	Solver methods
	"""

	def mainLoop(self):
		valid_actions = []

		# Number of moves made
		num_moves = 0

		while self._num_visited < const.NUM_TARGETS:
			# Can agent see a target, if so go towards it
			actions = self.isTargetVisible()
			if actions is not None:
				# Randomly select an action
				choice = random.randint(0, len(actions)-1)
				chosen_action = actions[choice]
			# Goto nearest unvisited location using voting table
			else:
				chosen_action = self._v_map.findUnvisitedDirectionNonStatic(*self._agent.getPos())

			# Perform the action
			self._agent.performAction(chosen_action)

			# Increment the move (time step) counter
			num_moves += 1

			# Iterate the DR handler
			self._dr.iterate()

			# Does this new position match a target position
			match = self.checkMatches()

			# Increment the counter if there's a match
			if match: 
				self._num_visited += 1

				# Inform the discovery rate handler
				self._dr.discovery()

			# Get the agent's position
			a_x, a_y = self._agent.getPos()

			# Update visit map
			self._v_map.iterate(a_x, a_y, match, 0)

			# Save the action choice
			valid_actions.append(chosen_action)

		# Indicate the DR to finish up, get the mean discovery rate for this episode
		mu_DT = self._dr.finish()

		return valid_actions, mu_DT

	# Uses agent & target coordinates to find out whether there are any targets currently
	# visible to the agent
	def isTargetVisible(self):
		# Get the current agent position
		a_x, a_y = self._agent.getPos()

		# Iterate over all targets
		for t in self._targets:
			# The current target is unvisited
			if not t.getVisited():
				# Get the current target position
				t_x, t_y = t.getPos()

				# Is this target within a 1 radius?
				if (a_x <= t_x + 1 and a_x >= t_x - 1 and
					a_y <= t_y + 1 and a_y >= t_y - 1)	 :
					return Utility.possibleActionsForAngle(a_x, a_y, t_x, t_y)

		return None

	def checkMatches(self):
		# Get the current agent position
		a_x, a_y = self._agent.getPos()

		for t in self._targets:
			if a_x == t._x and a_y == t._y and not t.getVisited():
				t.setVisited(True)
				return True
