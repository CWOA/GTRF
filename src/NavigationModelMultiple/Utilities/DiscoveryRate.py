#!/usr/bin/env python

import numpy as np

"""
Class is in charge of managing the generation of target discovery rate statistics
"""

class DiscoveryRate:
	# Class constructor
	def __init__(self):
		"""
		Class arguments
		"""


		"""
		Class attributes
		"""



	"""
	Class methods
	"""

	# Called at the beginning of each episode
	def reset(self):
		# List storing discovery rates throughout the episode
		self._DT = []

		# Reset the timesteps since discovery counter
		self._non_disc_ctr = 0

	# Called at every episode timestep
	#
	#	BE SURE TO CALL THIS BEFORE DISCOVERY
	#
	def iterate(self):
		# Increment the counter counting the number of timesteps since new target visitation
		self._non_disc_ctr += 1

	# Called whenever a new target is visited
	def discovery(self):
		# Compute discovery/time value
		DT_val = 1.0 / self._non_disc_ctr

		# Reset the move without discovery counter
		self._non_disc_ctr = 0

		# Add it to the list
		self._DT.append(DT_val)

	# Called at the end of each episode
	def finish(self):
		# Convert to numpy array
		DT_np = np.asarray(self._DT)

		# Return the mean and standard deviation
		return np.mean(DT_np)

	"""
	Getters
	"""


	"""
	Setters
	"""