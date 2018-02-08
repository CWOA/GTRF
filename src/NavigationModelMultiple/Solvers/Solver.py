#!/usr/bin/env python

import TreeSolver
import ClosestSolver
import SequenceSolver
import NaiveSolver
import MotionSolver
import Constants as const
from Utility import Utility

"""
Classes that inherit this superclass provide some form of solution to the problem of
navigating the given environment and visiting targets. Each subclass utilises its own
strategy in order to solve the problem
"""

class EpisodeSolver:
	# Class constructor
	def __init__(self, solver_method):
		"""
		Class attributes
		"""

		# The actual solver method we're going to use
		if solver_method == const.SEQUENCE_SOLVER:
			self._solver = SequenceSolver.SequenceSolver()
		elif solver_method == const.CLOSEST_SOLVER:
			self._solver = ClosestSolver.ClosestSolver()
		elif solver_method == const.TREE_SOLVER:
			self._solver = TreeSolver.TreeSolver()
		elif solver_method == const.NAIVE_SOLVER:
			self._solver = NaiveSolver.NaiveSolver()
		elif solver_method == const.MOTION_SOLVER:
			self._solver = MotionSolver.MotionSolver()
		else:
			Utility.die("Solver method not recognised", __file__)

	# Giving initial conditions to this episode to the solver
	def reset(self, agent, targets):
		self._solver.reset(agent, targets)

	# Solve the given episode using the designated method
	def solveEpisode(self):
		return self._solver.solve()

	# Get the next action to perform for the solved episode
	def getNextAction(self):
		return self._solver.nextAction()

# Entry method/unit testing
if __name__ == '__main__':
	pass
