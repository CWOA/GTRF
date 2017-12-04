#!/usr/bin/env python

"""
Constructs a globally-optimal solution in a lightweight fashion. Assumes that a global
solution visits all targets, finds the cost of all permutations of target sequences,
returning the best (this is a far better solution than TreeSolver and is essentially
equivalent)
"""

# Class inherits properties of Solver superclass
class SequenceSolver(Solver):
	# Class constructor
	def __init__(self):

# Entry method/unit testing
if __name__ == '__main__':
