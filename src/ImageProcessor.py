#!/usr/bin/env python

import cv2
import roslib
roslib.load_manifest('uav_id')
import rospy as ros
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class ImageProcessor:
	def __init__(self):
		# Subscriber and respective callback functions
		sub_image = ros.Subscriber('image', Image, self.imageCallback)

		# Publishers

		# Class attributes
		self._bridge = CvBridge()
		self._window_name = "ImageProcessor"

	def imageCallback(self, image_msg):
		try:
			cv_image = self._bridge.imgmsg_to_cv2(image_msg, "bgr8")
		except CvBridgeError as e:
			print e

		cv2.imshow(self._window_name, cv_image)
		cv2.waitKey(3)

# Entry method
if __name__ == '__main__':
	# Create a ImageProcessor object instance
	ip = ImageProcessor()

	# Initialise this node
	ros.init_node('image_processor')

	# Spin away
	try:
		ros.spin()
	except KeyboardInterrupt:
		print "ImageProcessor node shutting down"
	finally:
		cv2.destroyAllWindows()
