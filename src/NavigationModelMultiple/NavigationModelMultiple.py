#!/usr/bin/env python

import DNN
import Constants as const
from Utility import Utility
from FieldMap import FieldMap

"""
This class forms the principal entry point for selecting experimentation,
see the main function below
"""

# Generate training data, save it to file and use as training data for DNN training
# then evaluate on the best model yielded from cross-fold validation
def generateTrainTest(experiment_name, iterations, visualise, use_simulator):
	# Initialise FieldMap instance for training data generation and perform it
	train_fm = FieldMap(	True, 
							experiment_name, 
							visualise=visualise, 
							use_simulator=use_simulator, 
							save=True						)
	saved_to_path = train_fm.generateTrainingData(iterations)

	# Can comment the above two lines and uncomment the one below to just run data
	# generation and testing
	# saved_to_path = "/home/will/catkin_ws/src/uav_id/tflearn/ICIP2018/data/TRAINING_DATA_visitation_marked_TO.h5"
	# saved_to_path = "/home/will/catkin_ws/src/uav_id/tflearn/ICIP2018/data/TRAINING_DATA_moving_equidistant.h5"

	# Use this training data to initialise and train the dual input CNN
	dnn = DNN.DNNModel(use_simulator=use_simulator)
	best_model_path = dnn.trainModel(experiment_name, data_dir=saved_to_path)

	# best_model_path = "/home/will/catkin_ws/src/uav_id/tflearn/ICIP2018/models/visitation_marked_TO_2018-02-13_22:28:27_CROSS_VALIDATE_4.tflearn"
	# best_model_path = "/home/will/catkin_ws/src/uav_id/tflearn/ICIP2018/models/visitation_marked_GAUSSIAN_2018-02-18_17:53:40_CROSS_VALIDATE_4.tflearn"

	# Use the best model path to test
	test_fm = FieldMap(		False, 
							experiment_name, 
							visualise=visualise, 
							use_simulator=use_simulator, 
							model_path=best_model_path		)
	test_fm.startTestingEpisodes(iterations)

# Just generate training examples
def generateTrainingExamples(iterations, visualise, use_simulator, save_video):
	# Experiment parameters
	experiment_name = "video_test"

	fm = FieldMap(	True,
					experiment_name,
					visualise=visualise, 
					use_simulator=use_simulator, 
					save=True, 
					save_video=save_video 			)
	fm.generateTrainingData(iterations)

"""
Train model on synthesised data
DON'T EXECUTE VIA ROSLAUNCH (no need to do so, just launch via python/terminal)
"""
def trainModel(iterations, use_simulator):
	pass

# Testing trained model on real example/problem
def testModel(iterations, visualise, use_simulator):
	# Experiment parameters
	experiment_name = "motion_test_MS"

	fm = FieldMap(False, experiment_name, visualise=visualise, use_simulator=use_simulator)
	fm.startTestingEpisodes(iterations)

# Method for testing/comparing solver methods
def compareSolvers(iterations, exp_name, visualise):
	fm = FieldMap(True, exp_name, visualise=visualise, use_simulator=False, second_solver=True)
	fm.compareSolvers(iterations)

# Method for generating videos comparing the employed method, the globally-optimal solution
def generateVideoComparison(iterations, exp_name, visualise):
	# Base model directory
	base = Utility.getICIPModelDir()

	use_simulator = False

	# 1) Static equidistant grid
	# exp_name = "static_grid"
	# best_model_path = "{}/equidistant_SEQUENCE_2018-01-31_12:22:23_CROSS_VALIDATE_6.tflearn".format(base)
	
	# 2) Moving equidistant grid
	# exp_name = "moving_grid"
	# best_model_path = "{}/moving_equidistant_2018-02-20_20:21:22_CROSS_VALIDATE_5.tflearn".format(base)

	# 3) Gaussian
	# exp_name = "gaussian"
	# best_model_path = "{}/visitation_marked_GAUSSIAN_2018-02-18_17:53:40_CROSS_VALIDATE_4.tflearn".format(base)

	# 4) Random
	# exp_name = "random"
	# best_model_path = "{}/model_SEQUENCE_2017-12-15_15:51:08_CROSS_VALIDATE_4.tflearn".format(base)

	# 5) Random simulator
	use_simulator = True
	pause_for_user_input = True
	exp_name = "random_simulator"
	best_model_path = "{}/sequence_SIMULATOR_2018-01-24_13:40:00_CROSS_VALIDATE_2.tflearn".format(base)

	fm = FieldMap(	False, 
					exp_name, 
					visualise=visualise,
					save_video=True,
					use_simulator=use_simulator,
					model_path=best_model_path	)
	fm.generateVideos(iterations, pause_beforehand=pause_for_user_input)

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

	# Save frames to individual per-episode videos?
	save_video = const.SAVE_VIDEO

	"""
	Primary function calls
	"""

	# generateTrainTest("equidistant_marked", iterations, visualise, use_simulator)
	# generateTrainingExamples(iterations, visualise, use_simulator, save_video)
	# trainModel(iterations, use_simulator)
	# testModel(iterations, visualise, use_simulator)
	compareSolvers(iterations, "naive_solution" visualise)
	# generateVideoComparison(iterations, "", visualise)
