#!/usr/bin/env python

# Core libraries
import sys
sys.path.append('../')
from tqdm import tqdm
import numpy as np
import h5py
import tensorflow as tf
import tflearn
from sklearn.model_selection import train_test_split

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import Constants as const

"""
TBC properly
"""

class SplitStreamDNNModel:
	def __init__(self):
		pass

	# Graph definition
	def defineNetworkArchitecture(self):
		# Network 0 definition (IMAGE) -> AlexNet
		net0 = tflearn.input_data([		None, 
										3, 
										3, 
										3		])
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
		optimiser0 = tflearn.Momentum(learning_rate=0.001)
		net0 = fully_connected(net0, 4, activation='softmax')
		net0 = regression(	net0, 
							optimizer=optimiser0,
							loss='categorical_crossentropy'		)

		# Network 1 definition (VISIT MAP)
		net1 = tflearn.input_data([		None,
										10,
										10,
										1		])
		net1 = conv_2d(net1, 12, 3, activation='relu')
		net1 = max_pool_2d(net1, 3, strides=2)
		net1 = local_response_normalization(net1)
		net1 = fully_connected(net1, 1024, activation='tanh')
		optimiser1 = tflearn.Momentum(learning_rate=0.001)
		net1 = fully_connected(net1, 4, activation='softmax')
		net1 = regression(	net1, 
							optimizer=optimiser1,
							loss='categorical_crossentropy'		)

		# Merge the two optimisation layers
		net = tflearn.merge([net0, net1], mode="elemwise_sum")

		return net

	def computeClassificationAccuracy(self, X0_test, X1_test, Y_test):
		correct = 0
		num_test_instances = X0_test.shape[0]
		pbar = tqdm(total=num_test_instances)
		for i in range(num_test_instances):
			img = X0_test[ i,:,:,:]
			m = X1_test[i,:,:,0]

			prediction = self.testModelSingle(img, m)
			max_idx = np.argmax(prediction)
			choice_vec = np.zeros(4)
			choice_vec[max_idx] = 1

			if np.array_equal(choice_vec, Y_test[i,:]):
				correct += 1

			pbar.update()

		pbar.close()

		accuracy = (float(correct)/num_test_instances)*100

		return accuracy

	def testModelSingle(self, img, occ):
		np_img = np.zeros((1, 3, 3, const.NUM_CHANNELS))
		np_img[0,:,:,:] = img

		np_map = np.zeros((1, const.MAP_WIDTH, const.MAP_HEIGHT, 1))
		np_map[0,:,:,0] = occ

		return self._model.predict([np_img, np_map])

	def loadData(self):
		# Prepare and load data
		data_path = "/home/will/catkin_ws/src/uav_id/tflearn/ICIP2018/data/training_data_SEQUENCE.h5"
		dataset = h5py.File(data_path, 'r')

		# Extract the datasets contained within the file as numpy arrays, simple :)
		X0 = dataset['X0'][()]
		X1 = dataset['X1'][()]
		Y = dataset['Y'][()]

		X_temp = np.zeros((X1.shape[0], X1.shape[1], X1.shape[2], 1))
		X_temp[:,:,:,0] = X1
		X1 = X_temp

		# Reduce number of instances
		# s = X0.shape[0]/8
		# X0 = X0[0:s,:,:,:]
		# X1 = X1[0:s,:,:,:]
		# Y = Y[0:s,:]

		return train_test_split(X0, X1, Y, train_size=0.9, random_state=42)

	def trainModel(self):
		X0_train, X0_test, X1_train, X1_test, Y_train, Y_test = self.loadData()

		save_location = "/home/will/catkin_ws/src/uav_id/tflearn/ICIP2018/models/split_stream.tflearn"

		net = self.defineNetworkArchitecture()
		self._model = tflearn.DNN(net)
		self._model.fit([X0_train, X1_train], [Y_train, Y_train], show_metric=True, n_epoch=50, validation_set=([X0_test, X1_test], [Y_test, Y_test]))
		self._model.save(save_location)

		print self.computeClassificationAccuracy(X0_test, X1_test, Y_test)

		return save_location

	def loadModel(self, model_dir=None):
		net = self.defineNetworkArchitecture()
		self._model = tflearn.DNN(net)
		self._model.load(model_dir)

# Entry method/unit testing
if __name__ == '__main__':
	dnn = SplitStreamDNNModel()
	dnn.main()
