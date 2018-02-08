#!/usr/bin/env python


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

	def reset(self, agent, targets):
		# Objects for the agent and all targets
		self._agent = agent
		self._targets = targets

		# Number of target objects
		self._num_targets = len(targets)

		# Total number of possible solutions (one or more of which is globally optimal)
		# self._complexity = math.factorial(self._num_targets)

		# Reset the graph
		# self._graph = nx.DiGraph()

		# Unique identifier counter for each node
		self._id_ctr = 0

	def solve(self):
		# self.growTree()
		# self._actions = self.findBestSolutions()
		# return len(self._actions)
		pass

	def nextAction(self):
		# return self._actions.pop(0)
		return 'F'