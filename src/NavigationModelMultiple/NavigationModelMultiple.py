#!/usr/bin/env python

import Constants as const
from Utilities.Utility import Utility
from Core.FieldMap import FieldMap
from Algorithms import DNN

from Algorithms.SplitStreamDNN import SplitStreamDNNModel
import tensorflow as tf

"""
This class forms the principal entry point for selecting experimentation,
see the main function below
"""

def temp():
	dnn = SplitStreamDNNModel()
	dnn.trainModel()

	tf.reset_default_graph()
	del dnn

	best_model_path = "/home/will/catkin_ws/src/uav_id/tflearn/ICIP2018/models/split_stream.tflearn"

	test_fm = FieldMap(		False, 
							"split stream", 
							visualise=False, 
							use_simulator=False, 
							model_path=best_model_path		)
	test_fm.startTestingEpisodes(iterations)

# Generate training data, save it to file and use as training data for DNN training
# then evaluate on the best model yielded from cross-fold validation
def generateTrainTest(	experiment_name, 
						iterations, 
						visualise, 
						use_simulator 		):
	# Initialise FieldMap instance for training data generation and perform it
	# train_fm = FieldMap(	True, 
	# 						experiment_name, 
	# 						visualise=visualise, 
	# 						use_simulator=use_simulator, 
	# 						save=True						)
	# saved_to_path = train_fm.generateTrainingData(iterations)

	# Can comment the above two lines and uncomment the one below to just run data
	# generation and testing
	# saved_to_path = "/home/will/catkin_ws/src/uav_id/tflearn/ICIP2018/data/TRAINING_DATA_visitation_marked_TO.h5"
	saved_to_path = "/home/will/catkin_ws/src/uav_id/tflearn/ICIP2018/data/TRAINING_DATA_individual_motion_20k.h5"

	# Use this training data to initialise and train the dual input CNN
	dnn = DNN.DNNModel(use_simulator=use_simulator)
	best_model_path = dnn.trainModel(experiment_name, data_dir=saved_to_path)

	# best_model_path = "/home/will/catkin_ws/src/uav_id/tflearn/ICIP2018/models/visitation_marked_TO_2018-02-13_22:28:27_CROSS_VALIDATE_4.tflearn"
	# best_model_path = "/home/will/catkin_ws/src/uav_id/tflearn/ICIP2018/models/split_stream.tflearn"

	# Use the best model path to test
	# test_fm = FieldMap(		False, 
	# 						experiment_name, 
	# 						visualise=visualise, 
	# 						use_simulator=use_simulator, 
	# 						model_path=best_model_path		)
	# test_fm.startTestingEpisodes(iterations)

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
def testModel(	iterations, 
				exp_name, 
				visualise, 
				use_simulator, 
				model_path,
				dist_method,
				mark_visitation 	):
	# Create FieldMap object instance
	fm = FieldMap(	False, 
					exp_name, 
					visualise=visualise,
					use_simulator=use_simulator,
					model_path=model_path,
					dist_method=dist_method,
					mark_visitation=mark_visitation		)

	# Let's test!
	fm.startTestingEpisodes(iterations)

# Method for testing/comparing solver methods
def compareSolvers(iterations, exp_name, visualise):
	# Create FieldMap object instance
	fm = FieldMap(	True, 
					exp_name, 
					visualise=visualise, 
					use_simulator=False, 
					second_solver=True 		)

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

	# Object distribution method
	dist_method = const.OBJECT_DIST_METHOD

	"""
	Primary function calls
	"""

	generateTrainTest("individual_motion", iterations, visualise, use_simulator)
	# generateTrainTest("herd_dynamics", iterations, visualise, use_simulator)

	# generateTrainingExamples(iterations, visualise, use_simulator, save_video)
	# trainModel(iterations, use_simulator)
	# testModel(iterations, visualise, use_simulator)
	# compareSolvers(iterations, "naive_solution", visualise)
	# generateVideoComparison(iterations, "", visualise)

	# Testing function calls

	# # Base model directory
	# base = Utility.getICIPModelDir()

	# # CU (best fold: 5)
	# model_path = "{}/model_CLOSEST_2017-12-14_20:04:09_CROSS_VALIDATE_5.tflearn".format(base)
	# testModel(iterations, "closest_unvisited", visualise, False, model_path, dist_method, False)

	# # TO (best fold: 4)
	# model_path = "{}/model_SEQUENCE_2017-12-15_15:51:08_CROSS_VALIDATE_4.tflearn".format(base)
	# testModel(iterations, "target_ordering", visualise, False, model_path, dist_method, False)

	# # Static grid (best fold: 6)
	# dist_method = const.STAT_DIST
	# model_path = "{}/equidistant_SEQUENCE_2018-01-31_12:22:23_CROSS_VALIDATE_6.tflearn".format(base)
	# testModel(iterations, "static_grid", visualise, False, model_path, dist_method, False)

	# # Gaussian (best fold: 2)
	# dist_method = const.GAUS_DIST
	# model_path = "{}/gaussian_SEQUENCE_2018-01-31_16:11:20_CROSS_VALIDATE_2.tflearn".format(base)
	# testModel(iterations, "gaussian", visualise, False, model_path, dist_method, False)

	# # Random MARKED visitation (best fold: 4)
	# dist_method = const.PRNG_DIST
	# model_path = "{}/visitation_marked_TO_2018-02-13_22:28:27_CROSS_VALIDATE_4.tflearn".format(base)
	# testModel(iterations, "random_marked", visualise, False, model_path, dist_method, True)

	# # Gaussian MARKED visitation (best fold: 4)
	# dist_method = const.GAUS_DIST
	# model_path = "{}/visitation_marked_GAUSSIAN_2018-02-18_17:53:40_CROSS_VALIDATE_4.tflearn".format(base)
	# testModel(iterations, "gaussian_marked", visualise, False, model_path, dist_method, True)