#!/usr/bin/env python

"""
Greedy solution simply chooses the closest unvisited target and selects the best actions
towards navigating towards that target/individual
"""

# Class inherits properties of Solver superclass
class ClosestSolver:
	# Class constructor
	def __init__(self):
		"""
		Class attributes
		"""

	def reset(self, a_x, a_y, targets):
		# Set episode attributes
		self._a_x = a_x
		self._a_y = a_y
		self._targets = targets

	# Use technique/strategy to solve this episode
	def solveEpisode(self):
		pass

# Entry method/unit testing
if __name__ == '__main__':
	pass
