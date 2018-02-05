#!/usr/bin/env python

import DualInputCNN
import YourAlgorithm

"""
TBC
"""

class Algorithm:
	# Class consturctor
	def __init__(self, algorithm_method):
		"""
		Class attributes
		"""

		if algorithm_method == const.ALGORITHM_DUAL_INPUT_CNN:
			self._algorithm = DualInputCNN.DualInputCNN()
		elif algorithm_method == const.ALGORITHM_YOUR_ALGORITHM:
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

# Entry method/unit testing
if __name__ == '__main__':
	pass
