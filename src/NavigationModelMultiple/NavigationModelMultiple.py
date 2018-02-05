#!/usr/bin/env python

import Constants as const
from FieldMap import FieldMap

# Generating training data
def generateTrainingExamples(iterations, visualise, use_simulator):
	fm = FieldMap(visualise=visualise, use_simulator=use_simulator, save=True)
	fm.startTrainingEpisodes(iterations)

"""
Train model on synthesised data
DON'T EXECUTE VIA ROSLAUNCH (no need to do so, just launch via python/terminal)
"""
def trainModel(iterations, use_simulator):
	fm = FieldMap(use_simulator=use_simulator, training_model=True)
	fm.trainModel()

# Testing trained model on real example/problem
def testModel(iterations, visualise, use_simulator):
	fm = FieldMap(visualise=visualise, use_simulator=use_simulator)
	fm.startTestingEpisodes(iterations)

# Method for testing/comparing solver methods
def compareSolvers(iterations, visualise):
	fm = FieldMap(visualise=visualise, use_simulator=False, second_solver=True)
	fm.compareSolvers(iterations)

# Entry method
if __name__ == '__main__':
	"""
	Runtime argument definitions
	"""

	# Whether to visualise visual input/map via OpenCV imshow
	visualise = False

	# Whether or not to use ROS/Gazebo simulator for synthesised visual input
	use_simulator = False

	# Number of iterations/episodes to generate
	iterations = 10000

	"""
	Function calls
	"""

	# generateTrainingExamples(iterations, visualise, use_simulator)
	# trainModel(iterations, use_simulator)
	testModel(iterations, visualise, use_simulator)
	# compareSolvers(iterations, visualise)
