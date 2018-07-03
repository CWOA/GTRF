#!/usr/bin/env python

import TreeSolver
import ClosestSolver
import SequenceSolver
import NaiveSolver
import MotionSolver
import Constants as const

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

		# Which method to use?
		self._solver_method = solver_method

		# The actual solver method we're going to use
		if self._solver_method == const.SEQUENCE_SOLVER:
			self._solver = SequenceSolver.SequenceSolver()
		elif self._solver_method == const.CLOSEST_SOLVER:
			self._solver = ClosestSolver.ClosestSolver()
		elif self._solver_method == const.TREE_SOLVER:
			self._solver = TreeSolver.TreeSolver()
		elif self._solver_method == const.NAIVE_SOLVER:
			self._solver = NaiveSolver.NaiveSolver()
		elif self._solver_method == const.MOTION_SOLVER:
			self._solver = MotionSolver.MotionSolver()
		else:
			Utility.die("Solver method not recognised", __file__)

	# Giving initial conditions to this episode to the solver
	def reset(self, agent, targets, rand_pos=None):
		# If we're using the motion solver, give it the target positions versus time
		if self._solver_method == const.MOTION_SOLVER:
			self._solver.reset(agent, targets, rand_pos=rand_pos)
		# Otherwise, act as normal
		else:
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
