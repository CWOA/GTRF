#!/usr/bin/env python

# Core libraries
import sys
sys.path.append('../')
import cv2
import math
import random
import numpy as np
import Constants as const

# ROS libraries
import tf
import roslib
roslib.load_manifest('uav_id')
import rospy as ros
from gazebo_msgs.msg import *
from gazebo_msgs.srv import *
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge, CvBridgeError

# My libraries
from Utilities.Utility import Utility

class Visualiser:
	# Class constructor
	def __init__(self, use_simulator):
		"""
		Class attributes
		"""

		# Whether to generate visual input using ROS/gazebo simulator or not
		self._use_simulator = use_simulator
		if self._use_simulator:
			# Initialise an instance of the simulator bridge
			self._simulator_bridge = SimulatorBridge()

		# Dimensions for complete state display
		self._disp_width = const.MAP_WIDTH * const.GRID_PIXELS
		self._disp_height = const.MAP_HEIGHT * const.GRID_PIXELS

		print "Initialised Visualiser"

	# Informs simulation bridge to change model states to reflect new instance/episode
	def resetAgentTargets(self, a_x, a_y, target_poses):
		# Ensure we're meant to be here
		assert(self._use_simulator)

		# Indicate the change
		self._simulator_bridge.resetAgentTargets(a_x, a_y, target_poses)

	# Choose between rendering here (in gridworld) or rendering using ROS simulator
	def update(self, state, render_occ_map=False):
		# Create image to render to
		img = np.zeros((self._disp_height, self._disp_width, 3), np.uint8)

		# Get agent position
		a_x = state[0][0]
		a_y = state[0][1]

		# Render the occupancy grid map in grayscale if we're supposed to
		occ_map0 = None
		occ_map1 = None
		if render_occ_map:
			occ_map0, occ_map1 = self.renderOccupancyMap(state[2])

		# Set image to background colour
		img[:] = const.BACKGROUND_COLOUR

		# Render target locations
		for target in state[1]:
			# Round target positions to the nearest integer if motion is enabled
			t_x = int(round(target[0]))
			t_y = int(round(target[1]))

			img = self.renderGridPosition(t_x, t_y, img, const.TARGET_COLOUR)

		# Make a copy of the image (we don't want to render visitation history
		# to the agent subview)
		img_copy = img.copy()

		# Render target locations
		for target in state[1]:
			# Round target positions to the nearest integer if motion is enabled
			t_x = int(round(target[0]))
			t_y = int(round(target[1]))

			img = self.renderGridPosition(t_x, t_y, img, const.TARGET_COLOUR)

		if const.OCCUPANCY_MAP_MODE is not const.MOTION_MODE:
			# Render visited locations
			for x in range(const.MAP_WIDTH):
				for y in range(const.MAP_HEIGHT):
					# Have we been to this coordinate before?
					if state[2][y,x]:
						# Render this square as have being visited
						img = self.renderGridPosition(x, y, img, const.VISITED_COLOUR)

		# Render current agent position to both images
		img = self.renderGridPosition(		a_x, 
											a_y, 
											img, 
											const.AGENT_COLOUR		)
		img_copy = self.renderGridPosition(		a_x, 
												a_y, 
												img_copy, 
												const.AGENT_COLOUR		)

		# Number of pixels to pad subview with
		pad = 1

		# Use the ROS simulator to generate the subview
		if self._use_simulator:
			subview = self._simulator_bridge.renderSubviewUsingSimulator(a_x, a_y, state[1])

			# Downsample the image
			subview = cv2.resize(subview, (const.IMG_DOWNSAMPLED_WIDTH, const.IMG_DOWNSAMPLED_HEIGHT))
		else:
			# Pad the image with grid_pixels in background colour in case the agent
			# is at a border
			subview = self.padBorders(img_copy, pad)

			s_x = ((a_x + 1) * const.GRID_PIXELS) - pad
			s_y = ((a_y + 1) * const.GRID_PIXELS) - pad

			subview = subview[s_y:s_y+3*pad,s_x:s_x+3*pad]

			# Render the window that is visible to the agent
			# img = self.renderVisibilityWindow(	a_x,
			# 									a_y,
			# 									self._visible_padding,
			# 									self._line_thickness,
			# 									img,
			# 									self._visible_colour	)

		# Assign renders as class attributes
		self._render_img = img
		self._subview = subview

		return img, subview, occ_map0, occ_map1

	# Given the occupancy map in numpy array form, construct a grayscale image of it for
	# visualisation purposes
	def renderOccupancyMap(self, occ_map):
		# Check which occupancy map mode we're in
		if const.OCCUPANCY_MAP_MODE == const.VISITATION_MODE:
			img = np.zeros((const.MAP_HEIGHT, const.MAP_WIDTH, 3), np.uint8)
		elif const.OCCUPANCY_MAP_MODE == const.MOTION_MODE:
			img0 = np.zeros((const.MAP_HEIGHT, const.MAP_WIDTH, 3), np.uint8)
			img1 = np.zeros((const.MAP_HEIGHT, const.MAP_WIDTH, 3), np.uint8)

		# Make sure the sizes are the same
		assert(occ_map.shape[0] == const.MAP_WIDTH)
		assert(occ_map.shape[1] == const.MAP_HEIGHT)

		# Iterate over the occupancy map
		for i in range(const.MAP_WIDTH):
			for j in range(const.MAP_HEIGHT):
				# Check which occupancy map mode we're in
				if const.OCCUPANCY_MAP_MODE == const.VISITATION_MODE:
					# Get the current occupancy map value
					val = occ_map[j,i]

					# If the value is non-zero
					if val > 0:
						# Scale value to grayscale range
						val = Utility.scaleValueFromRangeToRange(	val,
																	const.UNVISITED_VAL,
																	const.AGENT_VAL,
																	120.0,
																	255.0				)

						# Assign scaled value to the image
						img[j,i,:] = val
				elif const.OCCUPANCY_MAP_MODE == const.MOTION_MODE:
					val0 = occ_map[j,i,0]
					val1 = occ_map[j,i,1]

					if val0 > 0:
						# Scale value to grayscale range
						val0 = Utility.scaleValueFromRangeToRange(	val0,
																	const.MOTION_EMPTY_VAL,
																	const.MOTION_HIGH_VALUE,
																	0.0,
																	255.0				)
						img0[j,i,:] = val0

					if val1 > 0:
						val1 = Utility.scaleValueFromRangeToRange(	val1,
																	const.MOTION_EMPTY_VAL,
																	const.MOTION_HIGH_VALUE,
																	0.0,
																	255.0				)
						img1[j,i,:] = val1

		if const.OCCUPANCY_MAP_MODE == const.VISITATION_MODE:
			return img, None
		elif const.OCCUPANCY_MAP_MODE == const.MOTION_MODE:
			return img0, img1

	def padBorders(self, img, pad):
		# Create a new image with the correct borders
		pad_img = np.zeros((self._disp_height+pad*2, self._disp_width+pad*2, 3), np.uint8)

		pad_img[:] = const.BACKGROUND_COLOUR

		# Copy the image to the padded image
		pad_img[pad:self._disp_height+pad,pad:self._disp_width+pad] = img

		return pad_img

	def renderGridPosition(self, x, y, img, colour):
		img[y*const.GRID_PIXELS:(y+1)*const.GRID_PIXELS,
			x*const.GRID_PIXELS:(x+1)*const.GRID_PIXELS,:] = colour

		return img

	# Display the two rendered images
	def display(self, wait_amount):
		cv2.imshow(const.MAIN_WINDOW_NAME, self._render_img)
		cv2.imshow(const.AGENT_WINDOW_NAME, self._subview)
		cv2.waitKey(wait_amount)

# Acts as an intermediary between visualisation requests and the ROS/gazebo simulator
class SimulatorBridge:
	# Class constructor
	def __init__(self):
		"""
		Class attributes
		"""

		# CV Bridge (converting ROS images to openCV)
		self._bridge = CvBridge()

		self._cow_angles = []

		"""
		ROS/Class attributes
		"""

		# Initialise this ROS node
		ros.init_node(const.VIS_ROS_NODE_NAME)

		# ROS subscribers
		self._sub_uav_img = ros.Subscriber(			const.UAV_CAM_IMG_TOPIC_NAME,
													Image,
													self.imageCallback					)

		# ROS Services

		# Wait for the required services to become available
		ros.wait_for_service(const.SET_MODEL_STATE_SERVICE_NAME)

		# Initialise services
		self._set_model_state = ros.ServiceProxy(	const.SET_MODEL_STATE_SERVICE_NAME, 
													SetModelState 						)

	# Converts a given position from a left-hand coordinate system to right-hand
	# i.e. top-left image reference frame to bottom-right gazebo
	def leftHandToRightHandPosition(self, x, y):
		return const.MAP_HEIGHT - y, const.MAP_WIDTH - x

	# Teleport the given model name to an absolute position in a 2D plane with angle heading
	def teleportAbsolute(self, model_name, x, y, z, yaw):
		# Convert positions to the correct coordinate system
		x, y = self.leftHandToRightHandPosition(x, y)

		# Convert RPY euler angle to quaternion
		quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)

		# Construct the gazebo model state message
		desired_pose = ModelState()
		desired_pose.model_name = model_name
		desired_pose.pose.position.x = x * const.SCALE_FACTOR
		desired_pose.pose.position.y = y * const.SCALE_FACTOR
		desired_pose.pose.position.z = z
		desired_pose.pose.orientation.x = quaternion[0]
		desired_pose.pose.orientation.y = quaternion[1]
		desired_pose.pose.orientation.z = quaternion[2]
		desired_pose.pose.orientation.w = quaternion[3]

		# Try sending message via service call
		try:
			_ = self._set_model_state(desired_pose)
		except ros.ServiceException as e:
			Utility.die("Service call failed for reason: {}".format(e), __file__)

	# Use the ROS simulator to generate the current agent subview
	def renderSubviewUsingSimulator(self, a_x, a_y, targets):
		# Move the agent to the updated location
		self.teleportAbsolute(const.ROBOT_NAME, a_x, a_y, const.DEFAULT_HEIGHT, 0)

		if const.INDIVIDUAL_MOTION:
			for i in range(len(targets)):
				# Construct current target model name
				target_name = const.BASE_TARGET_NAME + "_" + str(i)

				# Extract positions
				t_x = targets[i][0]
				t_y = targets[i][1]

				# Move the target
				self.teleportAbsolute(target_name, t_x, t_y, 0, self._cow_angles[i])

		# Convert agent to gazebo reference frame
		g_x, g_y = self.leftHandToRightHandPosition(a_x, a_y)

		# Sleep for lil bit (to allow the image callback to catchup)
		# NEED BETTER SOLUTION
		ros.sleep(0.1)

		return self._subview

	# Callback function for attached simulated camera
	def imageCallback(self, image_msg):
		# Try converting the image into OpenCV format from ROS
		try:
			# Attempt to convert
			self._subview = self._bridge.imgmsg_to_cv2(image_msg, "bgr8")
		except CvBridgeError as e:
			print e

	# Directly modifies gazebo model states to reflect randomly generated positions
	# for both the agent and all targets for this new training or testing instance/
	# episode
	def resetAgentTargets(self, a_x, a_y, target_poses):
		# Make a quick check
		assert(len(target_poses) == const.NUM_TARGETS)

		# Move the agent
		self.teleportAbsolute(const.ROBOT_NAME, a_x, a_y, const.DEFAULT_HEIGHT, 0)

		self._cow_angles = []

		# Move all the targets
		for i in range(len(target_poses)):
			# Construct current target model name
			target_name = const.BASE_TARGET_NAME + "_" + str(i)

			# Extract positions
			t_x = target_poses[i][0]
			t_y = target_poses[i][1]

			# Generate a random angle, convert to radians
			angle = math.radians(random.randint(0, 360))

			self._cow_angles.append(angle)

			# Move the target
			self.teleportAbsolute(target_name, t_x, t_y, 0, angle)

# Entry method for unit testing
if __name__ == '__main__':
	pass
