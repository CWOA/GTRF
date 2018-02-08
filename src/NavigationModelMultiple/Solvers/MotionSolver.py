#!/usr/bin/env python

import networkx as nx
import Constants as const

"""
TBC
"""

class MotionSolver:
	# Class constructor
	def __init__(self):
		"""
		Class attributes/properties
		"""

	"""
	Mandatory class methods
	"""

	def reset(self, agent, targets, rand_pos):
		# Objects for the agent and all targets
		self._agent = agent
		self._targets = targets

		# Number of target objects
		self._num_targets = len(targets)
		assert(self._num_targets == const.NUM_TARGETS)

		# The positions over time targets will take
		self._rand_pos = rand_pos

		# The max number of steps for those random positions
		self._num_steps = len(self._rand_pos)
		assert(self._num_steps == const.RANDOM_WALK_NUM_STEPS+1)

		self.printRandPos()

		# Total number of possible solutions (one or more of which is globally optimal)
		# self._complexity = math.factorial(self._num_targets)

		# Reset the graph
		self._graph = nx.DiGraph()

		# Unique identifier counter for each node
		self._id_ctr = 0

	def solve(self):
		self.growTree()
		self._actions = self.findBestSolutions()
		return len(self._actions)

	def nextAction(self):
		# return self._actions.pop(0)
		return 'F'

	"""
	Tree-growing methods
	"""

	def growTree(self):
		# Create the root node
		root = NodeAttribute()


	def growTreeRecursive(self, parent_attr):
		# Find targets to visit
		targets = parent_attr.possibleTargets()

		# Iterate over possible targets to visit
		for target in targets:
			

	"""
	Utility functions
	"""

	# Just print the random target position versus time matrix
	def printRandPos(self):
		for i in range(self._num_steps):
			print self._rand_pos[i]

class NodeAttribute:
	def __init__(	self,
					node_id,
						):

class EdgeAttribute:
	def __init__(	self	):

# Entry method/unit testing
if __name__ == '__main__':
	pass
