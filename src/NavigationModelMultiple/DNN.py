#!/usr/bin/env python

import tflearn
import numpy as np
import Constants as const
from Utility import Utility
from sklearn.model_selection import train_test_split
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

class DNNModel:
	# Class constructor
	def __init__(self):
		### Class attributes

		# All training/testing data
		self._data = []

		# Number of classes
		self._num_classes = len(const.ACTIONS)

		# Input data dimensions for IMAGE input
		self._img_width = const.GRID_PIXELS * 3
		self._img_height = const.GRID_PIXELS * 3

		# Network architecture
		self._network = self.defineDNN()

		# Model declaration
		self._model = tflearn.DNN(	self._network,
									tensorboard_verbose=0,
									tensorboard_dir=Utility.getTensorboardDir()	)


	# Load pickled data from file
	def loadData(self):
		# Load pickled data
		with open(self._FM._data_dir, 'rb') as fin:
			self._data = pickle.load(fin)

		num_instances = len(self._data)

		# Agent subview
		X0 = np.zeros((num_instances, self._img_width, self._img_height, const.NUM_CHANNELS))
		# Visitation map
		X1 = np.zeros((num_instances, const.MAP_WIDTH, const.MAP_HEIGHT, 1))

		# Ground truth labels
		Y = np.zeros((num_instances, self._num_classes))

		for i in range(num_instances):
			X0[i,:,:,:] = self._data[i][0]
			X1[i,:,:,0] = self._data[i][1]
			Y[i,:] = self._data[i][2]

		return self.segregateData(X0, X1, Y)

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

	def trainModel(self):
		# Get and split the data
		X0_train, X0_test, X1_train, X1_test, Y_train, Y_test = self.loadData()

		# Train the model
		self._model.fit(	[X0_train, X1_train],
							Y_train,
							validation_set=([X0_test, X1_test], Y_test),
							n_epoch=self._n_epochs,
							batch_size=64,
							run_id=self._model_name,
							show_metric=True								)

		self.loadSaveModel(load=False)

		self.evaluateModel(X0_test, X1_test, Y_test)

	def testModelSingle(self, img, visit_map):
		# Insert image into 4D numpy array
		np_img = np.zeros((1, self._img_width, self._img_height, const.NUM_CHANNELS))
		np_img[0,:,:,:] = img

		# Insert map into 4D numpy array
		np_map = np.zeros((1, const.MAP_WIDTH, const.MAP_HEIGHT, 1))
		np_map[0,:,:,0] = visit_map

		# Predict on given img and map
		return self._model.predict([np_img, np_map])

	def evaluateModel(self, X0_test, X1_test, Y_test):
		print self._model.evaluate([X0_test, X1_test], Y_test)

	def loadSaveModel(self, load=True):
		model_dir = Utility.getModelDir()

		if load:
			self._model.load(model_dir)
			string = "Loaded"
		else: 
			self._model.save(model_dir)
			string = "Saved"

		print "{} TFLearn model at directory:{}".format(string, model_dir)

	def defineDNN(self):
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

		# Network 1 definition (VISIT MAP)
		net1 = tflearn.input_data([		None,
										const.MAP_HEIGHT,
										const.MAP_WIDTH,
										1					])
		net1 = conv_2d(net1, 12, 3, activation='relu')
		net1 = max_pool_2d(net1, 3, strides=2)
		net1 = local_response_normalization(net1)
		net1 = fully_connected(net1, 1024, activation='tanh')

		# Merge the networks
		net = tflearn.merge([net0, net1], "concat", axis=1)
		net = fully_connected(net, self._num_classes, activation='softmax')
		net = regression(		net, 
								optimizer='momentum',
								loss='categorical_crossentropy',
								learning_rate=0.001						)

		return net

# Entry method for unit testing
if __name__ == '__main__':
	dnn = DNNModel()
	dnn.loadSaveModel()
