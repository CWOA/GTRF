#!/usr/bin/env python

import roslib
roslib.load_manifest('uav_id')
import rospy as ros
import numpy as np
from gazebo_msgs.msg import *
from std_srvs.srv import Empty
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class GridCameraController:
	def __init__(self):
		### ROS Publishers
		self._pos_pub = ros.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)

		### ROS Subscribers
		self._pos_sub = ros.Subscriber('/gazebo/model_states', ModelStates, self.poseCallback)
		self._sub_image = ros.Subscriber('cam_img', Image, self.imageCallback)

		### ROS Parameters
		self._granularity = ros.get_param("~granularity", 1)

		### Class attributes
		# Movement vectors
		self._forward	= []
		self._backward 	= []
		self._left 		= []
		self._right 	= []
		self._up 		= []
		self._down 		= []

		# Store the current object pose
		self._current_pose = ModelState()

		# Store the current simulated camera image
		self._current_img = np.empty(1)

		# Model robot name from gazebo/launch file/xacro definition
		self._rob_name = "sim_cam"

		### Pre-processing
		self.updateMovementVectors(self._granularity)

	def updateMovementVectors(self, gran):
		self._forward	= [gran,	0,		0]
		self._backward	= [-gran,	0,		0]
		self._left		= [0,	 gran,		0]
		self._right		= [0,	-gran,		0]
		self._up		= [0,		0,	 gran]
		self._down		= [0,		0,	-gran]

	def poseCallback(self, data):
		try:
			idx = data.name.index(self._rob_name)
			self._current_pose.model_name = data.name[idx]
			self._current_pose.pose = data.pose[idx]
			self._current_pose.twist = data.twist[idx]
			#print current_pose
		except:
			pass

	# Callback function for attached simulated camera
	def imageCallback(self, image_msg):
		# Try converting the image into OpenCV format from ROS
		try:
			self._current_img = self._bridge.imgmsg_to_cv2(image_msg, "bgr8")
		except CvBridgeError as e:
			print e

	def move(self, move_input):
		desired_pose = self._current_pose
		desired_pose.pose.position.x = self._current_pose.pose.position.x + move_input[0]
		desired_pose.pose.position.y = self._current_pose.pose.position.y + move_input[1]
		desired_pose.pose.position.z = self._current_pose.pose.position.z + move_input[2]
		self._pos_pub.publish(desired_pose)

# Entry method
if __name__ == '__main__':
	# Initialise this ROS node
	ros.init_node('grid_camera_controller')

	# Create an object instance
	gcc = GridCameraController()

	ros.sleep(4)

	# Main loop
	try:
		while not ros.is_shutdown():
			ros.loginfo("Moving forward")
			gcc.move(gcc._forward)
			ros.sleep(2)

			ros.loginfo("Moving backward")
			gcc.move(gcc._backward)
			ros.sleep(2)

			ros.loginfo("Moving up")
			gcc.move(gcc._up)
			ros.sleep(2)

			ros.loginfo("Moving down")
			gcc.move(gcc._down)
			ros.sleep(2)

	except ros.ROSInterruptException:
		pass
	finally:
		ros.loginfo("Experiment finished, shutting everything down..")
		ros.signal_shutdown("Experiment finished")
