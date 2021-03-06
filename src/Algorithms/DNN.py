#!/usr/bin/env python

# Core libraries
import sys
sys.path.append('../')
import cv2
import h5py
import numpy as np
import Constants as const
import datetime
from tqdm import tqdm

# Machine learning scikit
from sklearn.model_selection import train_test_split
from sklearn import cross_validation

# Deep learning/tensorflow
import tflearn
import tensorflow as tf
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# My libraries/classes
from Utilities.Utility import Utility

"""
Deep Neural Network class for training, testing and evaluation on new inputs
"""

class DNNModel:
	# Class constructor
	def __init__(	self, 
					use_simulator=False,
					split_into_dual_networks=False	):
		"""
		Class arguments
		"""

		# If we're using the ROS/gazebo simulator for visual input
		self._use_simluator = use_simulator

		# If the split input algorithm is enabled (for use as a baseline comparison)
		self._use_dual_networks = split_into_dual_networks

		"""
		Class attributes
		"""

		# Number of classes
		if const.USE_EXT_ACTIONS:
			self._num_classes = len(const.EXT_ACTIONS)
		else:
			self._num_classes = len(const.ACTIONS)

		# If we're using the ROS/gazebo simulator for visual input
		if use_simulator:
			self._img_width = const.IMG_DOWNSAMPLED_WIDTH
			self._img_height = const.IMG_DOWNSAMPLED_HEIGHT
		else:
			# Input data dimensions for IMAGE input
			self._img_width = const.GRID_PIXELS * 3
			self._img_height = const.GRID_PIXELS * 3

		# Results from training model and evaluation
		self._eval_results = {}

		"""
		Class setup
		"""

		# Reset TensorFlow DNN (required when doing multiple runs)
		tf.reset_default_graph()

		# If we're using the dual CNN with two optimisers
		if self._use_dual_networks:
			print "Utilising split stream CNN"
			self._network = self.defineDNN(dual_optimiser=True)
		else:
			self._network = self.defineDNN()

		# Model declaration
		self._model = tflearn.DNN(	self._network,
									tensorboard_dir=Utility.getTensorboardDir()	)

		print "Initialised DNN"

	# Load H5 data from file
	def loadData(self, data_dir):
		print "Loading data"

		# Load location
		if data_dir is None:
			# load_loc = Utility.getHDF5DataDir()
			load_loc = "{}/gaussian_SEQUENCE.h5".format(Utility.getDataDir())
		else:
			load_loc = data_dir

		# Load the data
		dataset = h5py.File(load_loc, 'r')

		print "Loaded data at: {}".format(load_loc)

		# Extract the datasets contained within the file as numpy arrays, simple :)
		X0 = dataset['X0'][()]
		X1 = dataset['X1'][()]
		Y = dataset['Y'][()]

		# Add extra dimension to X1 at the end if we're in visitation mode
		if const.OCCUPANCY_MAP_MODE == const.VISITATION_MODE:
			X_temp = np.zeros((X1.shape[0], X1.shape[1], X1.shape[2], 1))
			X_temp[:,:,:,0] = X1
			X1 = X_temp

		# Quarter the number of instances
		# s = X0.shape[0]/8
		# X0 = X0[0:s,:,:,:]
		# X1 = X1[0:s,:,:,:]
		# Y = Y[0:s,:]

		# Normalise all visual input from [0,255] to [0,1]
		# X0 = self.normaliseInstances(X0, 255)
		# X1 = self.normaliseInstances(X1, const.AGENT_VAL)

		print "Finished loading data"

		# Quickly check there are the same number of data instances
		assert(X0.shape[0] == X1.shape[0])
		assert(X1.shape[0] == Y.shape[0])

		return X0, X1, Y

	def segregateData(self, X0, X1, Y):
		# Split data into training/testing with the specified ratio
		X0_train, X0_test, X1_train, X1_test, Y_train, Y_test = train_test_split(	X0,
																					X1,
																					Y,
																					train_size=const.DATA_RATIO,
																					random_state=42					)

		# Print some info about what the data looks like
		print "X0_train.shape={:}".format(X0_train.shape)
		print "X0_test.shape={:}".format(X0_test.shape)
		print "X1_train.shape={:}".format(X1_train.shape)
		print "X1_test.shape={:}".format(X1_test.shape)
		print "Y_train.shape={:}".format(Y_train.shape)
		print "Y_test.shape={:}".format(Y_test.shape)

		return X0_train, X0_test, X1_train, X1_test, Y_train, Y_test

	# Construct a unique RunID (for tensorboard) for this training run
	def constructRunID(self, exp_name, fold_id=0):
		# Get a date string
		date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

		run_id = "{}_{}".format(exp_name, date_str)

		# If we're cross-validating, include the current fold id
		if const.CROSS_VALIDATE:
			run_id = "{}_CROSS_VALIDATE_{}".format(run_id, fold_id)

		return run_id

	# Train model and cross validate
	def crossValidate(self, exp_name, X0, X1, Y):
		print "{}-fold cross validation is enabled".format(const.NUM_FOLDS)

		# Extract the number of training instances (previously verified to be
		# consistent across training variables)
		num_instances = X0.shape[0]

		# Create the cross validation object that randomly shuffles too
		kf = cross_validation.KFold(num_instances, n_folds=const.NUM_FOLDS, shuffle=True)

		# Current fold number
		fold_number = 0

		# Dictionary of model save directories (key is fold number)
		model_save_dirs = {}

		# Iterate num folds times
		for train_idx, test_idx in kf:
			print "Beginning fold {}/{} complete.".format(fold_number+1, const.NUM_FOLDS)

			# Split the data for this fold
			X0_train, X0_test = X0[train_idx], X0[test_idx]
			X1_train, X1_test = X1[train_idx], X1[test_idx]
			Y_train, Y_test = Y[train_idx], Y[test_idx]

			# Construct a run_id
			run_id = self.constructRunID(exp_name, fold_id=fold_number)

			# Network architecture
			self._network = self.defineDNN()

			# Init the model
			self._model = tflearn.DNN(	self._network,
										tensorboard_verbose=0,
										tensorboard_dir=Utility.getTensorboardDir()	)

			# Train!
			if self._use_dual_networks:
				self._model.fit(	[X0_train, X1_train],
									[Y_train, Y_train],
									validation_set=([X0_test, X1_test], [Y_test, Y_test]),
									n_epoch=const.NUM_EPOCHS,
									batch_size=64,
									run_id=run_id,
									show_metric=True								)
			else:
				self._model.fit(	[X0_train, X1_train],
									Y_train,
									validation_set=([X0_test, X1_test], Y_test),
									n_epoch=const.NUM_EPOCHS,
									batch_size=64,
									run_id=run_id,
									show_metric=True								)

			# Save the trained model
			model_save_dirs[fold_number] = self.saveModel(run_id=run_id)

			# Evaluate
			self.evaluateModel(X0_test, X1_test, Y_test, fold_id=fold_number)

			# Delete variables/reset DNN
			tf.reset_default_graph()
			del self._network
			del self._model

			# Increment
			fold_number += 1

			print "Fold {}/{} complete.".format(fold_number, const.NUM_FOLDS)

		# Print all results
		print self._eval_results

		# Find the fold with the best classification result
		best_fold = max(self._eval_results.iterkeys(), key=(lambda key: self._eval_results[key]))

		print "Best fold = {}".format(best_fold)

		# Use this key to extract the directory of the best fold
		best_model_path = model_save_dirs[best_fold]

		# Report average and standard deviation classification results across all folds
		all_values = np.asarray(self._eval_results.values())

		print "Average accuracy = {}, standard deviation = {}".format(np.mean(all_values), np.std(all_values))

		return best_model_path

	# Complete function which loads the appropriate training data, trains the model,
	# saves it to file and evaluates the trained model's performance
	def trainModel(self, exp_name, data_dir=None):
		print "Training model"

		# Load the data
		X0, X1, Y = self.loadData(data_dir)

		# Sanity checking
		# self.inspectData(X0, X1, Y)

		tf.reset_default_graph()

		# If we're supposed to cross-validate results, do so
		if const.CROSS_VALIDATE:
			best_model_path = self.crossValidate(exp_name, X0, X1, Y)
		# Cross validation not enabled, just split, train and evaluate
		else:
			# Split the data into training/testing chunks
			X0_train, X0_test, X1_train, X1_test, Y_train, Y_test = self.segregateData(X0, X1, Y)

			# Run ID
			run_id = self.constructRunID(exp_name)

			# Train
			if self._use_dual_networks:
				self._model.fit(	[X0_train, X1_train],
									[Y_train, Y_train],
									validation_set=([X0_test, X1_test], [Y_test, Y_test]),
									n_epoch=const.NUM_EPOCHS,
									batch_size=64,
									run_id=run_id,
									show_metric=True								)
			else:
				self._model.fit(	[X0_train, X1_train],
									Y_train,
									validation_set=([X0_test, X1_test], Y_test),
									n_epoch=const.NUM_EPOCHS,
									batch_size=64,
									run_id=run_id,
									show_metric=True								)

			# self._model.evaluate([X0_test, X1_test], [Y_test, Y_test])

			# Evaluate how we did
			# self.evaluateModel(X0_test, X1_test, Y_test)

			# Save the trained model
			best_model_path = self.saveModel(run_id=run_id)

		return best_model_path

	# Just iteratively inspect the data so it appears to make sense (figuratively)
	def inspectData(self, X0, X1, Y):
		# Loop over all instances
		for i in range(X0.shape[0]):
			# Get the current image
			current_img = X0[i,:,:,:]

			# Normalise
			current_img /= 255

			# Display relevant things
			cv2.imshow(const.AGENT_WINDOW_NAME, current_img)
			print current_img
			print X1[i,:,:,:]
			print Utility.classVectorToAction(Y[i,:])

			# Wait for a keypress
			cv2.waitKey(0)

	def testModelSingle(self, img, visit_map):
		# Insert image into 4D numpy array
		np_img = np.zeros((1, self._img_width, self._img_height, const.NUM_CHANNELS))
		np_img[0,:,:,:] = img

		# If motion is enabled (for the occupancy grid), add a dimension
		if const.OCCUPANCY_MAP_MODE == const.VISITATION_MODE:
			np_map = np.zeros((1, const.MAP_WIDTH, const.MAP_HEIGHT, 1))
			np_map[0,:,:,0] = visit_map
		elif const.OCCUPANCY_MAP_MODE == const.MOTION_MODE:
			np_map = np.zeros((1, const.MAP_WIDTH, const.MAP_HEIGHT, 2))
			np_map[0,:,:,:] = visit_map
		else:
			Utility.die("Occupancy map mode not recognised in testModelSingle()", __file__)

		# Predict on given img and map
		return self._model.predict([np_img, np_map])

	def evaluateModel(self, X0_test, X1_test, Y_test, fold_id=0):
		if self._use_dual_networks:
			result = self._model.evaluate([X0_test, X1_test], [Y_test, Y_test])
		else:
			# Evaluate and get the result
			result = self._model.evaluate([X0_test, X1_test], Y_test)

		# Print it out
		print result

		# Save result to a {fold number: evaluation result} dict
		self._eval_results[fold_id] = result

	def loadModel(self, model_dir=None):
		if model_dir is None:
			pass

		print "Loading TFLearn model at directory:{}".format(model_dir)

		self._model.load(model_dir)

		print "Loaded model.".format(model_dir)

	def saveModel(self, run_id=None):
		if run_id is not None:
			model_dir = Utility.getModelDir()
			model_dir = "{}/{}.tflearn".format(model_dir, run_id)

		self._model.save(model_dir)

		print "Saved TFLearn model at directory:{}".format(model_dir)

		return model_dir

	# Normalise all given instances with a given value (255 in most cases)
	def normaliseInstances(self, array, value):
		for i in range(array.shape[0]):
			array[i,:,:,:] = array[i,:,:,:] / value

		return array

	"""
	Network architecture declaration methods
	"""

	# Defines dual input CNN architecture that takes both agent-centric inputs
	def defineDNN(self, dual_optimiser=False):
		# If motion is enabled (for the occupancy grid), add a dimension to input data
		if const.OCCUPANCY_MAP_MODE == const.VISITATION_MODE:
			visit_map_dims = 1
		elif const.OCCUPANCY_MAP_MODE == const.MOTION_MODE:
			visit_map_dims = 2
		else:
			Utility.die("Occupancy map mode not recognised in defineDNN()", __file__)

		# Network 0 definition (IMAGE) -> AlexNet
		net0 = tflearn.input_data([		None, 
										self._img_height, 
										self._img_width, 
										const.NUM_CHANNELS		])
		net0 = conv_2d(net0, 96, 11, strides=4, activation='relu')
		net0 = max_pool_2d(net0, 3, strides=2)
		net0 = local_response_normalization(net0)
		net0 = conv_2d(net0, 256, 5, activation='relu')
		net0 = max_pool_2d(net0, 3, strides=2)
		net0 = local_response_normalization(net0)
		net0 = conv_2d(net0, 384, 3, activation='relu')
		net0 = conv_2d(net0, 384, 3, activation='relu')
		net0 = conv_2d(net0, 256, 3, activation='relu')
		net0 = max_pool_2d(net0, 3, strides=2)
		net0 = local_response_normalization(net0)
		net0 = fully_connected(net0, 4096, activation='tanh')
		net0 = dropout(net0, 0.5)
		net0 = fully_connected(net0, 4096, activation='tanh')
		net0 = dropout(net0, 0.5)

		# If we should have dual optimisation regression layers(one for each agent input)
		if dual_optimiser:
			optimiser0 = tflearn.Momentum(learning_rate=0.001)
			net0 = fully_connected(net0, self._num_classes, activation='softmax')
			net0 = regression(	net0, 
								optimizer=optimiser0,
								loss='categorical_crossentropy'		)

		# Network 1 definition (VISIT MAP)
		net1 = tflearn.input_data([		None,
										const.MAP_HEIGHT,
										const.MAP_WIDTH,
										visit_map_dims		])
		net1 = conv_2d(net1, 12, 3, activation='relu')
		net1 = max_pool_2d(net1, 3, strides=2)
		net1 = local_response_normalization(net1)
		net1 = fully_connected(net1, 1024, activation='tanh')

		# If we should have dual optimisation regression layers(one for each agent input)
		if dual_optimiser:
			optimiser1 = tflearn.Momentum(learning_rate=0.001)
			net1 = fully_connected(net1, self._num_classes, activation='softmax')
			net1 = regression(	net1, 
								optimizer=optimiser1,
								loss='categorical_crossentropy'		)

			# Merge the two optimisation layers
			net = tflearn.merge([net0, net1], mode="elemwise_sum")

		# We should merge the networks and have a single optimisation layer
		else:
			# Merge the networks
			net = tflearn.merge([net0, net1], "concat", axis=1)

			# Softmax layer
			net = fully_connected(net, self._num_classes, activation='softmax')

			# Optimiser definition
			# optimiser = tflearn.Adam(learning_rate=0.001, beta1=0.99)
			# optimiser = tflearn.Momentum(learning_rate=0.001, lr_decay=0.96, decay_step=100)
			optimiser = tflearn.Momentum(learning_rate=0.001)

			# Regression layer
			net = regression(	net, 
								optimizer=optimiser,
								loss='categorical_crossentropy'		)

		return net

	# This function implements the training strategy that Tilo suggested
	def motionTrainingStrategy(self, data_dir):
		# Reset tensorflow graph
		tf.reset_default_graph()

		# Load all ground-truth training data
		X0, X1, Y = self.loadData(data_dir)

		# Split the data into training/testing chunks
		X0_train, X0_test, X1_train, X1_test, Y_train, Y_test = self.segregateData(X0, X1, Y)

		# Run identifier (for tensorboard)
		run_id = "training_strategy_motion_60k_V"

		# Callback class
		trainingCallback = TrainingCallback()

		# Training parameters
		pre_epochs = 10 # epochs to run prior to generating fake labels
		fake_epochs = 2 # epochs to run with fake labels
		norm_epochs = 2 # epochs to run with normal labels after fake
		alt_num = 4 # number of times to alternate between fake/normal
		post_epochs = 5 # epochs to run post fake label training
		repeat_spikes = 4

		invert_fake_labels = True

		# Initial run
		self._model.fit(	[X0_train, X1_train], 
							Y_train,
							validation_set=([X0_test, X1_test], Y_test),
							n_epoch=pre_epochs,
							batch_size=64,
							show_metric=True,
							run_id=run_id,
							callbacks=trainingCallback							)

		print "Finished initial training run"

		for x in range(repeat_spikes):
			# Number of alternate optimisation steps
			for i in range(alt_num):
				pbar = tqdm(total=X0_train.shape[0])

				Y_fake = np.zeros(Y_train.shape)
				# Generate labels on training data with current model state
				for j in range(X0_train.shape[0]):
					img_input = np.zeros((1, self._img_width, self._img_height, const.NUM_CHANNELS))
					img_input[0,:,:,:] = X0_train[j,:,:,:]
					map_input = np.zeros((1, const.MAP_WIDTH, const.MAP_HEIGHT, 2))
					map_input[0,:,:,:] = X1_train[j,:,:,:]

					prediction = self._model.predict([img_input, map_input])

					max_idx = np.argmax(prediction)
					choice_vec = np.zeros(len(const.EXT_ACTIONS))
					choice_vec[max_idx] = 1

					# Invert the label
					if invert_fake_labels:
						choice_vec = (1 - choice_vec) / (len(choice_vec) - 1.0)

					Y_fake[j,:] = choice_vec

					pbar.update()

				pbar.close()

				print "Finished generating fake labels"

				# Train on fake labels
				self._model.fit(	[X0_train, X1_train],
									Y_fake,
									validation_set=([X0_test, X1_test], Y_test),
									n_epoch=fake_epochs,
									batch_size=64,
									show_metric=True,
									run_id=run_id,
									callbacks=trainingCallback							)

				print "Finished training on fake labels"

				# Train on normal labels
				self._model.fit(	[X0_train, X1_train],
									Y_train,
									validation_set=([X0_test, X1_test], Y_test),
									n_epoch=norm_epochs,
									batch_size=64,
									show_metric=True,
									run_id=run_id,
									callbacks=trainingCallback							)

			print "Finished alternate optimisation"

			# Train on normal labels
			self._model.fit(	[X0_train, X1_train],
								Y_train,
								validation_set=([X0_test, X1_test], Y_test),
								n_epoch=post_epochs,
								batch_size=64,
								show_metric=True,
								run_id=run_id,
								callbacks=trainingCallback							)

		print "Finished training completely"

		# self.evaluateModel(X0_test, X1_test, Y_test)

		self.saveModel(run_id=run_id)


class TrainingCallback(tflearn.callbacks.Callback):
	def __init__(self):
		self._train_acc = []

	def on_batch_end(self, training_state, snapshot=False):
		# print "Training acc is: {}".format(training_state.global_acc)
		# self._train_acc.append(training_state.global_acc)
		pass

# Entry method for unit testing
if __name__ == '__main__':
	pass
