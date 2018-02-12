#!/usr/bin/env python

import DNN
import Constants as const
from FieldMap import FieldMap

"""
This class forms the principal entry point for selecting experimentation,
see the main function below
"""

# Generate training data, save it to file and use as training data for DNN training
# then evaluate on the best model yielded from cross-fold validation
def generateTrainTest(iterations, use_simulator):
	# Experiment parameters
	experiment_name = "visitation_marked_TO"

	# Initialise FieldMap instance for training data generation and perform it
	train_fm = FieldMap(True, visualise=False, use_simulator=use_simulator, save=True)
	saved_to_path = train_fm.generateTrainingData(iterations, exp_name=experiment_name)

	# Use this training data to initialise and train the dual input CNN
	dnn = DNN.DNNModel(use_simulator=use_simulator)
	best_model_path = dnn.trainModel(experiment_name, data_dir=saved_to_path)

	# Use the best model path to test
	test_fm = FieldMap(False, visualise=False, use_simulator=use_simulator, model_path=best_model_path)
	test_fm.startTestingEpisodes(iterations, experiment_name)

# Generating training data
def generateTrainingExamples(iterations, visualise, use_simulator):
	fm = FieldMap(True, visualise=visualise, use_simulator=use_simulator, save=True)
	fm.generateTrainingData(iterations)

"""
Train model on synthesised data
DON'T EXECUTE VIA ROSLAUNCH (no need to do so, just launch via python/terminal)
"""
def trainModel(iterations, use_simulator):
	pass

# Testing trained model on real example/problem
def testModel(iterations, visualise, use_simulator):
	fm = FieldMap(False, visualise=visualise, use_simulator=use_simulator)
	fm.startTestingEpisodes(iterations)

# Method for testing/comparing solver methods
def compareSolvers(iterations, visualise):
	fm = FieldMap(True, visualise=visualise, use_simulator=False, second_solver=True)
	fm.compareSolvers(iterations)

# Entry method
if __name__ == '__main__':
	"""
	Function calls

	Constant arguments to functions can be overidden here, by default run-time
	arguments are located at the top of the "Constants.py" file
	"""

	# Whether to visualise visual input/map via OpenCV imshow for debugging purposes
	visualise = const.VISUALISE

	# Whether or not to use ROS/Gazebo simulator for synthesised visual input
	use_simulator = const.USE_SIMULATOR

	# Number of episodes to test on or generate training examples
	iterations = const.ITERATIONS

	"""
	Primary function calls
	"""

	# generateTrainTest(iterations, use_simulator)
	generateTrainingExamples(iterations, visualise, use_simulator)
	# trainModel(iterations, use_simulator)
	# testModel(iterations, visualise, use_simulator)
	# compareSolvers(iterations, visualise)
