#!/usr/bin/env python

import os
import cv2
import math
import roslib
import random
import pickle
roslib.load_manifest('uav_id')
import rospy as ros
import numpy as np
from gazebo_msgs.msg import *
from std_srvs.srv import Empty
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from NavigationModel import NavigationModel

class GridCameraController:
	# Class initalised with desired starting position for simulated UAV. Default is (0,0,5)
	def __init__(self, init_x=0, init_y=0, init_z=5):
		### ROS Publishers
		self._pos_pub = ros.Publisher('set_model_state', ModelState, queue_size=1)

		### ROS Subscribers
		self._pos_sub = ros.Subscriber('get_all_model_states', ModelStates, self.poseCallback)
		self._sub_image = ros.Subscriber('cam_img', Image, self.imageCallback)

		### ROS Parameters
		self._granularity = ros.get_param("~granularity", 1)
		self._disp_img = ros.get_param("~disp_raw_img_feed", False)
		self._store_history = ros.get_param("~store_history", False)
		self._save_location = ros.get_param("~history_save_loc", "")

		### Class attributes
		# Movement vectors
		self._forward			= []	# +x
		self._backward 			= []	# -x
		self._left 				= []	# +y
		self._right 			= []	# -y
		self._up 				= []	# +z
		self._down 				= []	# -z
		# Diagonal movement vectors (for when z is disabled/fixed?)
		self._forward_left		= []	# +x-y
		self._forward_right		= []	# +x+y
		self._backward_left		= []	# -x-y
		self._backward_right	= []	# -x+y

		# Boundaries for cattle visiblity at fixed z=5 and camera resolution=640x480
		self._x_bound = 3.5
		self._y_bound = 4.5

		# If enabled, the history of frames/movements will be stored
		self._history = []

		# Store the current object pose
		self._current_pose = ModelState()

		# Store the current simulated camera image
		self._current_img = np.empty(1)

		# Model robot name from gazebo/launch file/xacro definition
		self._robot_name = "sim_cam"

		# For converting ROS image topics to OpenCV format
		self._bridge = CvBridge()

		# OpenCV window name 
		self._window_name = "UAV Image feed"

		self._nav_model = NavigationModel(load_model=True)

		### Pre-processing
		# Update movement granularity given the supplied parameter
		self.updateMovementVectors(self._granularity)

		# Teleport the UAV to the given initial position
		ros.sleep(1)
		ros.loginfo("Initialising grid UAV with x:{:}, y:{:}, z:{:}".format(init_x, init_y, init_z))
		self.teleportAbsolute(init_x, init_y, init_z)

	### ROS callback functions
	def poseCallback(self, data):
		try:
			idx = data.name.index(self._robot_name)
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

		# If we're meant to display image we've receivd
		if self._disp_img:
			cv2.imshow(self._window_name, self._current_img)
			cv2.waitKey(3)

	### Class methods
	def updateMovementVectors(self, gran):
		self._forward	= [gran,	0,		0]
		self._backward	= [-gran,	0,		0]
		self._left		= [0,	 gran,		0]
		self._right		= [0,	-gran,		0]
		self._up		= [0,		0,	 gran]
		self._down		= [0,		0,	-gran]
		# Diagonal movement vectors (for when z is disabled/fixed)
		self._forward_left		= [gran,	-gran,		0]
		self._forward_right		= [gran,	 gran,		0]
		self._backward_left		= [-gran,	-gran,		0]
		self._backward_right	= [-gran,	 gran,		0]

	# Teleport the UAV to an absolute position, only used for initial UAV position
	def teleportAbsolute(self, x, y, z):
		desired_pose = ModelState()
		desired_pose.model_name = self._robot_name
		desired_pose.pose.position.x = x
		desired_pose.pose.position.y = y
		desired_pose.pose.position.z = z
		desired_pose.pose.orientation.x = 0
		desired_pose.pose.orientation.y = 0
		desired_pose.pose.orientation.z = 0
		desired_pose.pose.orientation.w = 1
		self._pos_pub.publish(desired_pose)

	# Teleport the UAV relative to its current position by some movement vector
	def teleportRelative(self, move_input):
		if self._store_history:
			self._history.append((self._current_img, move_input))

		desired_pose = self._current_pose
		desired_pose.pose.position.x = self._current_pose.pose.position.x + move_input[0]
		desired_pose.pose.position.y = self._current_pose.pose.position.y + move_input[1]
		desired_pose.pose.position.z = self._current_pose.pose.position.z + move_input[2]
		self._pos_pub.publish(desired_pose)

	def moveForward(self):
		self.teleportRelative(self._forward)

	def moveBackward(self):
		self.teleportRelative(self._backward)

	def moveLeft(self):
		self.teleportRelative(self._left)

	def moveRight(self):
		self.teleportRelative(self._right)

	def moveUp(self):
		self.teleportRelative(self._up)

	def moveDown(self):
		self.teleportRelative(self._down)

	def moveForwardLeft(self):
		self.teleportRelative(self._forward_left)

	def moveForwardRight(self):
		self.teleportRelative(self._forward_right)

	def moveBackwardLeft(self):
		self.teleportRelative(self._backward_left)

	def moveBackwardRight(self):
		self.teleportRelative(self._backward_right)

	# Randomly generate a position within the boundaries
	def generatePositionInBounds(self, z_height=5):
		x_pos = random.uniform(self._x_bound, -self._x_bound)
		y_pos = random.uniform(self._y_bound, -self._y_bound)
		z_pos = z_height

		return x_pos, y_pos, z_pos

	def produceTrainingInstance(self):
		x_pos, y_pos, z_pos = self.generatePositionInBounds(z_height=5)

		# Teleport UAV to generated position
		self.teleportAbsolute(x_pos, y_pos, z_pos)

		# Ground truth movement vector
		# ["do nothing",forward,backward,left,right]
		movement_vector = [0,0,0,0,0]

		# If generated position is within "do nothing" bounds +/-1
		if x_pos <= 1 and x_pos >= -1 and y_pos <= 1 and y_pos >= -1:
			movement_vector = [1,0,0,0,0]
		else:
			# Get relative position
			rel_x = 0 - x_pos
			rel_y = 0 - y_pos

			# Compute angle to origin (centre of cow)
			angle = math.atan2(rel_y, rel_x)

			# Move forward
			if angle < math.pi/4 and angle > -math.pi/4:
				movement_vector = [0,1,0,0,0]
				print "Moving forward"
			# Move left
			elif angle >= math.pi/4 and angle < 3*math.pi/4:
				movement_vector = [0,0,0,1,0]
				print "Moving left"
			# Move right
			elif angle <= math.pi/4 and angle > -3*math.pi/4:
				movement_vector = [0,0,0,0,1]
				print "Moving right"
			# Move backward
			elif angle >= 3*math.pi/4 or angle <= -3*math.pi/4:
				movement_vector = [0,0,1,0,0]
				print "Moving backward"
			# Error!
			else:
				print "Error, oops"

		# Sleep for a bit (to allow image callback to update)
		ros.sleep(0.1)

		# Save the history
		self._history.append((self._current_img, movement_vector))

	def runTestingInstance(self):
		# Generate a random starting position
		x_pos, y_pos, z_pos = self.generatePositionInBounds(z_height=5)

		# Teleport UAV to generated position
		self.teleportAbsolute(x_pos, y_pos, z_pos)

		# Sleep for a bit (to allow image callback to update)
		ros.sleep(0.1)

		centred = False

		#self._nav_model.evaluateTrainedModel()

		raw_input()

		while not centred:
			movement = self._nav_model.predictOnFrame(self._current_img)[0]

			print movement

			ind_max = np.argmax(movement)

			# Do nothing
			if ind_max == 0:
				centred = True
				print "Centred, done!"
				break
			elif ind_max == 1: 
				self.moveForward()
				print "Moved forward"
			elif ind_max == 2:
				self.moveBackward()
				print "Moved backward"
			elif ind_max == 3: 
				self.moveLeft()
				print "Moved left"
			elif ind_max == 4: 
				self.moveRight()
				print "Moved right"

			# Sleep for a bit (to allow image callback to update)
			ros.sleep(0.2)

			#raw_input()


	# Called once main ROS loop terminates (naturally or via keyboard interrupt)
	def cleanUp(self):
		# Store movement history if we're meant to
		if self._store_history:
			# Pickle history
			# with open(self._save_location, "wb") as fout:
			# 	pickle.dump(self._history, fout)
			# 	ros.loginfo("Saved history to: {:}".format(self._save_location))

			# Save history into image folers
			directory = "/home/will/catkin_ws/src/uav_id/output/nav_data/"
			ctr = 0
			num_classes = 5

			for i in range(num_classes):
				path = directory+str(i)+"/"
				if not os.path.exists(path):
					os.makedirs(path)

			for item in self._history:
				for i in range(num_classes):
					if item[1][i]:
						file = directory + str(i) + "/" + str(ctr) + ".jpg"
						cv2.imwrite(file, item[0])
						ctr += 1

# Entry method
if __name__ == '__main__':
	# Initialise this ROS node
	ros.init_node('grid_camera_controller')

	# Some constants/parameters
	testing = True
	number_of_instances = 1000

	# Create object instance
	gcc = GridCameraController()

	ros.sleep(4)

	# Main loop
	try:
		# while not ros.is_shutdown():
		# 	ros.loginfo("Moving forward")
		# 	gcc.moveForward()
		# 	ros.sleep(2)

		# 	ros.loginfo("Moving backward")
		# 	gcc.moveBackward()
		# 	ros.sleep(2)

		# 	ros.loginfo("Moving up")
		# 	gcc.moveUp()
		# 	ros.sleep(2)

		# 	ros.loginfo("Moving down")
		# 	gcc.moveDown()
		# 	ros.sleep(2)

		# 	ros.loginfo("Moving left")
		# 	gcc.moveLeft()
		# 	ros.sleep(2)

		# 	ros.loginfo("Moving right")
		# 	gcc.moveRight()
			# ros.sleep(2)

		for i in range(0, number_of_instances):
			if testing:
				gcc.runTestingInstance()
			else:
				gcc.produceTrainingInstance()

			print "Progress = {}/{}".format(i+1, number_of_instances)
			raw_input()

	except ros.ROSInterruptException:
		pass
	finally:
		gcc.cleanUp()
		ros.loginfo("Experiment finished, shutting everything down..")
		ros.signal_shutdown("Experiment finished")
