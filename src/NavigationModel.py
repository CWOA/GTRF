#!/usr/bin/env python

import os
import cv2
import h5py
import pickle
import tflearn
import numpy as np
from sklearn.model_selection import train_test_split
import tflearn
from tflearn.data_utils import image_preloader
from tflearn.data_utils import build_hdf5_image_dataset
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

class NavigationModel:
    # Initialise this class
    def __init__(self, load_model=False):
        ### Constant definitions

        # Base directory for TFLearn attributes
        self._base_dir = "/home/will/catkin_ws/src/uav_id/tflearn"

        # Directories/paths
        data_dir = '../tflearn/data/navigation_data.pkl'

        # Location of folder-based data
        self._folder_data_dir = os.path.join(self._base_dir, "data/nav_data/")

        # Filename for hdf5-based dataset
        self._hdf5_data_dir = os.path.join(self._base_dir, "data/dataset.h5")

        tensorboard_dir = '../tflearn/tensorboard/'
        checkpoint_dir = '../tflearn/checkpoints/navigation_model'

        # Directory to save or load TFLearn model
        self._model_dir = os.path.join(self._base_dir, "models/navigation_model.tflearn")

        # Number of possible classes (do nothing, forward, backward, left, right)
        self._num_classes = 5

        # Dimensions to resize images (training or testing) to
        self._resize_width = 160
        self._resize_height = 120

        # Number of image channels
        self._num_channels = 3

        # How verbose tensorboard is (0: fastest, 3: most detail)
        self._tensorboard_verbose = 0

        ### Pre-processing/intialisation

        # AlexNet declaration
        self._network = self.AlexNet(   self._num_classes, 
                                        self._resize_width, 
                                        self._resize_height, 
                                        self._num_channels      )

        # Model declaration
        self._model = tflearn.DNN(      self._network, 
                                        checkpoint_path=checkpoint_dir, 
                                        max_checkpoints=3,
                                        tensorboard_verbose=self._tensorboard_verbose, 
                                        tensorboard_dir=tensorboard_dir                 )

        # Should we load the model?
        if load_model:
            self.loadModel()

    # AlexNet network definition
    def AlexNet(self, num_classes, resize_width, resize_height, num_channels):
        network = input_data(shape=[None, resize_height, resize_width, num_channels])
        network = conv_2d(network, 96, 11, strides=4, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = conv_2d(network, 256, 5, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = conv_2d(network, 384, 3, activation='relu')
        network = conv_2d(network, 384, 3, activation='relu')
        network = conv_2d(network, 256, 3, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = fully_connected(network, 4096, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 4096, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, num_classes, activation='softmax')
        network = regression(network, optimizer='momentum',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
        return network

    # Use hdf5 loader
    def loadhdf5Data(self):
        # Load the file
        h5f = h5py.File(self._hdf5_data_dir, 'r')

        # Fetch the arrays
        X = h5f['X']
        Y = h5f['Y']

        # Convert to numpy
        X = X[()]
        Y = Y[()]

        return X, Y

    # Use tflearn image preloader to fetch data
    def loadFolderData(self):
        X, Y = image_preloader(     self._folder_data_dir, 
                                    image_shape=(self._resize_width, self._resize_height), 
                                    mode='folder',
                                    categorical_labels=True,                               )
        # Convert to numpy array
        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    # Load data from pickle file in (image, class_vector) tuple form
    def loadPickleData(self, data_dir):
        print "Preparing training and testing datasets"
        with open(data_dir, 'rb') as fin:
            instances = pickle.load(fin)

        num_instances = len(instances)

        print "Found {} instances".format(num_instances)

        X = np.zeros((num_instances, resize_height, resize_width, num_channels))
        Y = np.zeros((num_instances, num_classes))

        for i in range(len(instances)):
            X[i,:,:,:] = cv2.resize(instances[i][0], (self._resize_width, self._resize_height))
            Y[i,:] = np.array(instances[i][1])

        return X, Y

    # Load data from file, segregate into training/testing
    def loadData(self, load_method=0):
        # Load hdf5 dataset
        if load_method == 0:
            X, Y = self.loadhdf5Data()
        # Load the data from a folder using the built in TFLearn function
        elif load_method == 1:
            X, Y = self.loadFolderData()
        # Use pickled data file
        elif load_method == 2:
            X, Y = self.loadPickleData()
        else:
            print "Loade method not recognised, breaking"
            return

        # Split data into training/testing
        X_train, X_test, Y_train, Y_test = train_test_split(    X, 
                                                                Y, 
                                                                test_size=0.1, 
                                                                random_state=42 )

        # Print some info about what the training data looks like
        print "X_train.shape={:}".format(X_train.shape)
        print "X_test.shape={:}".format(X_test.shape)
        print "Y_train.shape={:}".format(Y_train.shape)
        print "Y_test.shape={:}".format(Y_test.shape)

        return X_train, X_test, Y_train, Y_test

    # Builds hdf5 dataset for fast retrieval using tflearn, only needs to be called once
    def buildhdf5Data(self):
        build_hdf5_image_dataset(   target_path=self._folder_data_dir,
                                    image_shape=(self._resize_width, self._resize_height),
                                    output_path=self._hdf5_data_dir,
                                    mode='folder',
                                    categorical_labels=True                               )

    # Train a TFLearn/AlexNet model
    def trainModel(self):
        # Get the data
        X_train, X_test, Y_train, Y_test = self.loadData()

        # Train the network
        self._model.fit(    X_train, 
                            Y_train, 
                            n_epoch=100, 
                            validation_set=0.1, 
                            shuffle=True,
                            show_metric=True, 
                            batch_size=64, 
                            snapshot_step=200,
                            snapshot_epoch=False, 
                            run_id='navigation_model'   )

        # Save out the model
        self._model.save(self._model_dir)

        # Evaluate performance on testing data
        print self._model.evaluate(X_test, Y_test)

    # Load a trained model + data and evaluate performance
    def evaluateTrainedModel(self):
        # Load a model we've already trained
        self.loadModel()

        # Load up some testing data
        _, X, _, Y = self.loadData()

        # Evaluate performance
        performance = self._model.evaluate(X, Y)
        print "\n\nModel performance = {}\n\n".format(performance)

    # Load the model from a TFLearn checkpoint
    def loadModel(self):
        self._model.load(self._model_dir)

        print "Loaded TFLearn model at directory:{}".format(self._model_dir)

    # Given an image and a loaded model, predict the image's class
    def predictOnFrame(self, frame):
        # Resize image to the expected size
        frame = cv2.resize(frame, (self._resize_width, self._resize_height))

        # Insert into 4D array
        np_frame = np.zeros((1, self._resize_height, self._resize_width, self._num_channels))
        np_frame[0,:,:,:] = frame

        # Normalise the image
        np_frame = np.divide(np_frame, 255.0)

        # Predict on this frame
        return self._model.predict(np_frame)

# Main entry method
if __name__ == '__main__':
    NM = NavigationModel(load_model=True)

    # NM.buildhdf5Data()

    # NM.trainModel()

    _, X, _, Y = NM.loadData(0)

    print NM._model.evaluate(X, Y)

    cv2.namedWindow("test")
    for i in range(100):
        #cv2.imshow("test", X[i,:,:,:])

        pred = NM.predictOnFrame(X[i,:,:,:])[0]

        print "Ground truth vector:{:}".format(Y[i,:])
        print "Prediction: {:}".format(pred)
        print "Pred: {:}".format(np.argmax(pred))

        cv2.waitKey()
