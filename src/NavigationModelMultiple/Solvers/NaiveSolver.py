#!/usr/bin/env python

import time
import copy
import math
import random
import networkx as nx
import Constants as const
import VisitationMap
from Utility import Utility
import matplotlib.pyplot as plt

"""
This class is for "attempting" to solve an episode algorithmically to provide a baseline
for the ICIP paper
details on how it operates:
"""


class NaiveSolver:
	# Class constructor
	def __init__(self):
		self._v_map = VisitationMap.MapHandler()

	"""
	Mandatory class methods
	"""

	def reset(self, agent, targets):
		# Objects for the agent and all targets
		self._agent = copy.deepcopy(agent)
		self._targets = copy.deepcopy(targets)

		# Number of target objects
		self._num_targets = len(targets)

		self._v_map.reset(*self._agent.getPos())

		self._num_visited = 0

	def solve(self):
		self._actions = self.mainLoop()
		return len(self._actions)

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
				chosen_action = self._v_map.findUnvisitedDirection(*self._agent.getPos())

			# Perform the action
			self._agent.performAction(chosen_action)

			# Increment the move (time step) counter
			num_moves += 1

			# Check
			if self.checkMatches():
				self._num_visited += 1

			# Update visit map
			self._v_map.update(*self._agent.getPos())

			# Save the action choice
			valid_actions.append(chosen_action)

		return valid_actions

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
