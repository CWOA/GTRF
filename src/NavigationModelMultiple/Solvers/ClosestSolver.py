#!/usr/bin/env python

from Utility import Utility

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

	"""
	Mandatory class methods
	"""

	def reset(self, agent, targets):
		# Objects for the agent and all targets
		self._agent = agent
		self._targets = targets

		# Number of target objects
		self._num_targets = len(targets)

		# Action sequence for a solution
		self._actions = []

	# Use technique/strategy to solve this episode instance
	def solve(self):
		actions = []

		# Loop until we've visited all targets
		while not self.allTargetsVisited():
			# Get current position of the agent
			a_x, a_y = self._agent.getPos()

			# Find coordinates for the closest unvisited target
			_, c = self.findClosestTarget()

			# Find best action sequence between the agent and the current closest unvisited
			a = Utility.actionSequenceBetweenCoordinates(a_x, a_y, c[0], c[1])

			# Insert at the end of the actions list
			actions.append(a)

			# The agent is now at the closest unvisited
			self._agent.setPos(c[0], c[1])

			# Mark the target as having been visited
			self.markTargetVisited(c[0], c[1])

		# Flatten out the list
		self._actions = [i for s in actions for i in s]

		return len(self._actions)

	def nextAction(self):
		return self._actions.pop(0)

	"""
	Class methods
	"""

	# Find the target with coordinates equal to those given, mark it as visited
	def markTargetVisited(self, x, y):
		for target in self._targets:
			if target.getPosTuple() == (x, y):
				target.setVisited(True)

	# Returns True if all targets have been visited
	def allTargetsVisited(self):
		for target in self._targets:
			if not target.getVisited():
				return False

		return True

	# Returns the coordinates of the closest target to the current agent position that
	# hasn't already been visited
	def findClosestTarget(self):
		# Get the current agent position
		a_x, a_y = self._agent.getPos()

		best_dist = float("inf")
		best_coords = (-1,-1)

		# Iterate over each target
		for target in self._targets:
			# Check that we haven't already visited this target
			if not target.getVisited():
				# Find the distance
				distance = Utility.distanceBetweenPoints((a_x, a_y), target.getPosTuple())

				# Is the current distance better
				if distance < best_dist:
					best_dist = distance
					best_coords = target

		return (a_x, a_y), best_coords.getPosTuple()

# Entry method/unit testing
if __name__ == '__main__':
	pass
