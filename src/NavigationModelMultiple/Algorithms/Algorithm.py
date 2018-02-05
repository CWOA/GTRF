#!/usr/bin/env python

import DualInputCNN
import YourAlgorithm
import Constants as const

"""
TBC
"""

class Algorithm:
	# Class consturctor
	def __init__(	self, 
					algorithm_method, 
					use_simulator		):
		"""
		Class attributes
		"""

		# The chosen algorithm
		self._algorithm_selection = algorithm_method

		if self._algorithm_selection == const.ALGORITHM_DUAL_INPUT_CNN:
			self._algorithm = DualInputCNN.DualInputCNN(use_simulator)
		elif self._algorithm_selection == const.ALGORITHM_YOUR_ALGORITHM:
			self._algorithm = YourAlgorithm.YourAlgorithm()
		else:
			Utility.die("Algorithm method not recognised")

	# Signal the respective algorithm to reset
	def reset(self):
		self._algorithm.reset()

	# Step one iteration
	# Input:
	#		(1): Agent-centric visual field of the environment
	#		(2): Occpancy grid map
	# Output:
	#		(1): Action selected by the respective algorithm
	def iterate(self, image, occupancy_map):
		return self._algorithm.iterate(image, occupancy_map)

	# If the selected algorithm method uses the loop detector, retrieve the 
	# amount of times it was triggered
	def getNumLoops(self):
		if self._algorithm_selection == const.ALGORITHM_DUAL_INPUT_CNN:
			return self._algorithm.getNumLoops()
		# Just return 0 if the selected algorithm doesn't have loop detection
		else:
			return 0

# Entry method/unit testing
if __name__ == '__main__':
	pass