#!/usr/bin/env python

from FieldMap import FieldMap

# Generating training data
def generateTrainingExamples(visualise, use_simulator):
	fm = FieldMap(visualise=visualise, use_simulator=use_simulator, save=True)
	fm.startTrainingEpisodes(20000)

"""
Train model on synthesised data
DON'T EXECUTE VIA ROSLAUNCH (no need to do so, just launch via python/terminal)
"""
def trainModel(use_simulator):
	fm = FieldMap(use_simulator=use_simulator, training_model=True)
	fm.trainModel()

# Testing trained model on real example/problem
def testModel(visualise, use_simulator):
	fm = FieldMap(visualise=visualise, use_simulator=use_simulator)
	fm.startTestingEpisodes(1000)

# Entry method
if __name__ == '__main__':
	"""
	Runtime argument definitions
	"""

	# Whether to visualise visual input/map via OpenCV imshow
	visualise = True

	# Whether or not to use ROS/Gazebo simulator for synthesised visual input
	use_simulator = True


	"""
	Function calls
	"""

	# generateTrainingExamples(visualise, use_simulator)
	# trainModel(use_simulator)
	testModel(visualise, use_simulator)
