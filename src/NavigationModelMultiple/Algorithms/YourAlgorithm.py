#!/usr/bin/env python

import random
import Constants as const

"""
Your algorithm goes here! (rename as necessary)

TBC description
"""

class YourAlgorithm:
	# Class constructor
	def __init__(self):
		"""
		Class attributes
		"""


		"""
		Class initialisation
		"""

	# Reset function is called each time a new episode is started
	def reset(self):
		pass

	# Step one iteration for an episode (decide where to move based on input)
	# Input:
	#		(1): Agent-centric visual field of the environment
	#		(2): Occpancy grid map
	# Output:
	#		(1): Action selected by the respective algorithm
	def iterate(self, image, occupancy_map):

		#########################################################
		# YOUR ALGORITHM GOES HERE ##############################
		#########################################################

		# Just randomly select an action
		action = const.ACTIONS[random.randint(0, len(const.ACTIONS)-1)]

		return action

# Entry method/unit testing
if __name__ == '__main__':
	pass