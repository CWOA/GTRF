#!/usr/bin/env python

import cv2
import roslib
roslib.load_manifest('uav_id')
import rospy as ros
import numpy as np

import tensorflow as tf

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class ImageProcessor:
	def __init__(self):
		# Subscribers and respective callback functions ################################
		sub_image = ros.Subscriber('cam_img', Image, self.imageCallback)

		# Publishers ###################################################################

		# ROS parameters (from launch file or otherwise) ###############################

		# Defines the relative location of the Inception v3 tensorflow graph (.pb)
		self._v3_network_loc = ros.get_param("~v3-network-loc")

		# If true, input frames from the camera are saved via OpenCV at a defined
		# destination
		self._save_input_frames = ros.get_param("~save_input_frames")

		# Class attributes
		self._bridge = CvBridge()
		self._window_name = "ImageProcessor"

		# Tensorflow attributes/initialisation
		with tf.gfile.FastGFile(self._v3_network_loc, 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			_ = tf.import_graph_def(graph_def, name='')
			ros.loginfo("Loaded network at directory: %s", self._v3_network_loc)

		self._tf_sesh = tf.Session()
		self._pool_tensor = self._tf_sesh.graph.get_tensor_by_name('pool_3:0')

	# Called each time an image from the camera is received
	def imageCallback(self, image_msg):
		# Try converting the image into OpenCV format from ROS
		try:
			cv_image = self._bridge.imgmsg_to_cv2(image_msg, "bgr8")
		except CvBridgeError as e:
			print e

		# Check whether we should save the image out to file
		if self._save_input_frames:
			cv2.imwrite('')

		# Extract Inception v3 convolutional representation of the input image
		v3_out = self.extractV3Pool(cv_image)
		print v3_out
		print v3_out.shape

		# Display the image
		cv2.imshow(self._window_name, cv_image)
		cv2.waitKey(3)

	# Given an OpenCV image, extract its Inception V3 feature representation
	def extractV3Pool(self, image):
		# Change the image resolution, change into TF-style image
		image = cv2.resize(image, (299, 299), interpolation=cv2.INTER_CUBIC)
		tf_image = np.asarray(image)
		tf_image = np.expand_dims(tf_image, axis=0)

		# Get the image's pool3 CNN representation in 2048-wide vector form
		cnn_representation = self._tf_sesh.run(self._pool_tensor,
			{'Mul:0': tf_image})
		return cnn_representation[0][0][0]

# Entry method
if __name__ == '__main__':
	# Initialise this node
	ros.init_node('image_processor')

	# Create a ImageProcessor object instance
	ip = ImageProcessor()

	# Spin away
	try:
		ros.spin()
	except KeyboardInterrupt:
		print "ImageProcessor node shutting down"
	finally:
		cv2.destroyAllWindows()
		ip._tf_sesh.close()
