#!/usr/bin/env python

import os
import cv2
import math
import pickle
import random
import tflearn
import numpy as np
from collections import deque
from sklearn.model_selection import train_test_split
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

class dnn_model:
	# Class constructor
	def __init__(self, field_map):
		### Class attributes

		# Name of model
		self._model_name = "visit_map_navigation_model"

		# Fieldmap object to control it
		self._FM = field_map

		# All training/testing data
		self._data = []

		# Training data ratio (the rest is testing data)
		self._data_ratio = 0.9

		# Number of epochs to train for
		self._n_epochs = 40

		# Number of classes
		self._num_classes = len(self._FM._actions)

		# Input data dimensions for IMAGE input
		self._img_width = self._FM._grid_pixels * 3
		self._img_height = self._FM._grid_pixels * 3
		self._num_channels = 3

		# Input data dimensions for MAP input
		self._map_width = self._FM._grid_width
		self._map_height = self._FM._grid_height

		# Directories
		self._base_dir = self._FM._base_dir
		self._tensorboard_dir = os.path.join(self._base_dir, "tensorboard")
		self._model_dir = os.path.join(self._base_dir, "models/{}.tflearn".format(self._model_name))

		# Network architecture
		self._network = self.defineDNN()

		# Model declaration
		self._model = tflearn.DNN(	self._network,
									tensorboard_verbose=0,
									tensorboard_dir=self._tensorboard_dir	)


	# Load pickled data from file
	def loadData(self):
		# Load pickled data
		with open(self._FM._data_dir, 'rb') as fin:
			self._data = pickle.load(fin)

		num_instances = len(self._data)

		# Agent subview
		X0 = np.zeros((num_instances, self._img_width, self._img_height, self._num_channels))
		# Visitation map
		X1 = np.zeros((num_instances, self._map_width, self._map_height, 1))

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
																					train_size=self._data_ratio,
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

		self.saveModel()

		self.evaluateModel(X0_test, X1_test, Y_test)

	def loadModel(self):
		self._model.load(self._model_dir)

		print "Loaded TFLearn model at directory:{}".format(self._model_dir)

	def testModelSingle(self, img, visit_map):
		# Insert image into 4D numpy array
		np_img = np.zeros((1, self._img_width, self._img_height, self._num_channels))
		np_img[0,:,:,:] = img

		# Insert map into 4D numpy array
		np_map = np.zeros((1, self._map_width, self._map_height, 1))
		np_map[0,:,:,0] = visit_map

		# Predict on given img and map
		return self._model.predict([np_img, np_map])

	def evaluateModel(self, X0_test, X1_test, Y_test):
		# Evaluate performance
		print self._model.evaluate([X0_test, X1_test], Y_test)

	def saveModel(self):
		# Save out the trained model
		self._model.save(self._model_dir)

		print "Saved TFLearn model at directory:{}".format(self._model_dir)

	def defineDNN(self):
		# Network 0 definition (IMAGE) -> AlexNet
		net0 = tflearn.input_data([		None, 
										self._img_height, 
										self._img_width, 
										self._num_channels		])
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
										self._map_height,
										self._map_width,
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

	def testModelOnRealExample(self):
		print "Testing model on real examples"

		# Load the model from file
		self.loadModel()

		# Sanity check
		# X0_train, X0_test, X1_train, X1_test, Y_train, Y_test = self.loadData()
		# self.evaluateModel(X0_test, X1_test, Y_test)

		# Object to detect infinite agent loops
		detector = LoopDetector()

		# Number of test/examples to run in total
		num_examples = 10

		for i in range(num_examples):
			# Reset the grid
			self._FM.reset()

			# Reset the detector
			detector.reset()

			# Number of targets the agent has visited
			num_visited = 0

			# Number of moves the agent has made
			num_moves = 0

			while num_visited != self._FM._num_targets:
				# Get the map
				visit_map = self._FM._map.copy()

				# Render the updated view
				render_img, subview = self._FM.render()

				# Mark the current location of the agent
				visit_map[self._FM._agent_y, self._FM._agent_x] = 10

				# Based on this state, use the model to predict where to go
				prediction = self.testModelSingle(subview, visit_map)

				# Find the index of the max argument
				max_idx = np.argmax(prediction)
				choice = np.zeros(self._num_classes)
				choice[max_idx] = 1
				action = self._FM.classVectorToAction(choice)

				# Add the suggested action and check history
				if detector.addCheckAction(action):
					print "DETECTED INFINITE AGENT LOOP"

				# Make the move
				num_visited += self._FM.performAction(action)

				# Increment the number of moves made by the agent
				num_moves += 1

				# Display the image
				cv2.imshow(self._FM._window_name, render_img)
				cv2.imshow(self._FM._window_name_agent, subview)
				# print action
				# print visit_map
				cv2.waitKey(0)

			# Print some stats
			print "Solving this example took {} steps".format(num_moves)

# Class designed to help with detecting whether the agent is stuck in an infinite loop
class LoopDetector:
	# Class constructor
	def __init__(self, max_queue_size=3):
		# Start fresh
		self.reset()

		# Maximum length of queue
		self._max_queue_size = max_queue_size

	# Reset so we can start a new instance
	def reset(self):
		# Queue to store past actions
		self._actions = deque()

	# Add an action and check the queue
	def addCheckAction(self, action):
		self.addActionToQueue(action)
		return self.checkForLoop()

	# Add an action to the queue
	def addActionToQueue(self, action):
		# Add the action
		self._actions.append(action)

		# Check the length of the queue
		if len(self._actions) == self._max_queue_size + 1:
			# We need to pop an older entry
			self._actions.popleft()

	# Check whether the supplied sequence and 
	def checkActionSequence(self, sequence):
		equal = True

		if len(sequence) == len(self._actions):
			for i in range(len(sequence)):
				if sequence[i] != self._actions[i]:
					equal = False
					break
		else:
			return False

		return equal

	# Given the current action queue, detect whether a loop has occurred
	def checkForLoop(self):
		if self.checkActionSequence(['F', 'B', 'F']): return True
		if self.checkActionSequence(['B', 'F', 'B']): return True
		if self.checkActionSequence(['L', 'R', 'L']): return True
		if self.checkActionSequence(['R', 'L', 'R']): return True

		return False

class FieldMap:
	# Class constructor
	def __init__(		self, 
						visualise=False, 
						agent_global_view=True, 
						random_agent_pos=True, 
						save=False						):
		### Class attributes

		# Data directories
		self._base_dir = "/home/will/catkin_ws/src/uav_id/tflearn"
		self._data_dir = os.path.join(self._base_dir, "data/multiple_nav_data.pkl")

		# Whether or not we should be saving output to file
		self._save_output = save

		# Training data list to pickle upon completion
		self._training_output = []

		# Which training data generation method to use
		# see begin() function
		self._agent_has_global_view = agent_global_view

		# Dimensions of grid
		self._grid_width = 10
		self._grid_height = 10

		# Position of agent should be generated randomly
		self._random_agent_pos = random_agent_pos

		# Unit to move agent by each movement
		self._move_dist = 1

		# Number of target objects (number of cows)
		self._num_targets = 5

		# Possible agent actions
		self._actions = ['F', 'B', 'L', 'R']

		# Initialise all the necessary elements
		self.reset()

		### Attributes for agent-visible window

		# Number of cells centred around the agent visible around the agent
		# e.g. with padding = 1, visible window would be 3x3
		self._visible_padding = 1

		self._line_thickness = 5

		### Display attributes

		# Bool to decide whether to actually visualise
		self._visualise = visualise

		# Number of pixels a grid takes up
		self._grid_pixels = 1

		# Dimensions of display/visualisation grid
		self._disp_width = self._grid_pixels * self._grid_width
		self._disp_height = self._grid_pixels * self._grid_height

		# Window name
		self._window_name = "Visualisation grid"
		self._window_name_agent = "Agent subview"

		# Colours (BGR)
		self._background_colour = (42,42,23)
		self._visited_colour = (181,161,62)
		self._agent_colour = (89,174,110)
		self._target_colour = (64,30,162)
		self._visible_colour = (247,242,236)

	def render(self):
		# Create image to render to
		img = np.zeros((self._disp_height, self._disp_width, 3), np.uint8)

		# Get agent position
		a_x = self._agent_x
		a_y = self._agent_y

		# Set image to background colour
		img[:] = self._background_colour

		# Render target locations
		for target in self._targets:
			img = self.renderGridPosition(target[0], target[1], img, self._target_colour)

		# Make a copy of the image (we don't want to render visitation history
		# to the agent subview)
		img_copy = img.copy()

		# Render visited locations
		for x in range(self._grid_width):
			for y in range(self._grid_height):
				# Have we been to this coordinate before?
				if self._map[y,x]:
					# Render this square as have being visited
					img = self.renderGridPosition(x, y, img, self._visited_colour)

		# Render current agent position to both images
		img = self.renderGridPosition(		a_x, 
											a_y, 
											img, 
											self._agent_colour		)
		img_copy = self.renderGridPosition(		a_x, 
												a_y, 
												img_copy, 
												self._agent_colour		)

		# Number of pixels to pad subview with
		pad = self._grid_pixels * self._visible_padding

		# Pad the image with grid_pixels in background colour in case the agent
		# is at a border
		subview = self.padBorders(img_copy, pad)

		s_x = ((a_x + self._visible_padding) * self._grid_pixels) - pad
		s_y = ((a_y + self._visible_padding) * self._grid_pixels) - pad

		subview = subview[s_y:s_y+3*pad,s_x:s_x+3*pad]

		# Render the window that is visible to the agent
		# img = self.renderVisibilityWindow(	a_x,
		# 									a_y,
		# 									self._visible_padding,
		# 									self._line_thickness,
		# 									img,
		# 									self._visible_colour	)

		return img, subview

	def padBorders(self, img, pad):
		# Create a new image with the correct borders
		pad_img = np.zeros((self._disp_height+pad*2, self._disp_width+pad*2, 3), np.uint8)

		pad_img[:] = self._background_colour

		# Copy the image to the padded image
		pad_img[pad:self._disp_height+pad,pad:self._disp_width+pad] = img

		return pad_img

	def renderGridPosition(self, x, y, img, colour):
		img[y*self._grid_pixels:(y+1)*self._grid_pixels,
			x*self._grid_pixels:(x+1)*self._grid_pixels,:] = colour

		return img

	def renderVisibilityWindow(self, x, y, pad, thickness, img, colour):
		p1 = ((x-pad)*self._grid_pixels, (y-pad)*self._grid_pixels)
		p2 = ((x+pad+1)*self._grid_pixels, (y-pad)*self._grid_pixels)
		p3 = ((x+pad+1)*self._grid_pixels, (y+pad+1)*self._grid_pixels)
		p4 = ((x-pad)*self._grid_pixels, (y+pad+1)*self._grid_pixels)

		cv2.line(img, p1, p2, colour, thickness)
		cv2.line(img, p2, p3, colour, thickness)
		cv2.line(img, p3, p4, colour, thickness)
		cv2.line(img, p4, p1, colour, thickness)

		return img

	def initTargets(self):
		# Initialise the list of targets
		targets = []

		# Number of valid target positions generated
		num_generated = 0

		while num_generated < self._num_targets:
			# Generate a random position
			x_pos = random.randint(0, self._grid_width-1)
			y_pos = random.randint(0, self._grid_height-1)

			# Check the position isn't the same as the agent's position
			if x_pos != self._agent_x and y_pos != self._agent_y:
				# Generated position is valid
				valid_generation = True

				# Check the position isn't the same as any of the other targets
				for target in targets:
					if x_pos == target[0] and y_pos == target[1]:
						valid_generation = False

				# Randomly generated position is valid
				if valid_generation:
					targets.append((x_pos, y_pos))
					num_generated += 1

		return targets

	# Examines whether a target is currently visible and suggests actions towards it
	# NOTE
	# For the moment, this only works when visible padding is 1 (need a better solution)
	def checkVisibility(self, actions):
		a_x = self._agent_x
		a_y = self._agent_y

		desired_actions = []

		# Iterate over each target
		for target in self._targets:
			# Get the current target's position
			t_x = target[0]
			t_y = target[1]

			# 1: Top left
			if t_x == a_x - 1 and t_y == a_y - 1:
				desired_actions = ['F', 'L']
			# 2: Top middle
			elif t_x == a_x and t_y == a_y - 1:
				desired_actions = ['F']
			# 3: Top right
			elif t_x == a_x + 1 and t_y == a_y - 1:
				desired_actions = ['F', 'R']
			# 4: Middle left
			elif t_x == a_x - 1 and t_y == a_y:
				desired_actions = ['L']
			# 5: Middle right
			elif t_x == a_x + 1 and t_y == a_y:
				desired_actions = ['R']
			# 6: Bottom left
			elif t_x == a_x - 1 and t_y == a_y + 1:
				desired_actions = ['B', 'L']
			#7: Bottom middle
			elif t_x == a_x and t_y == a_y + 1:
				desired_actions = ['B']
			#8: Bottom right
			elif t_x == a_x + 1 and t_y == a_y + 1:
				desired_actions = ['B', 'R']

		# See which desired and possible actions are common (if any)
		possible_actions = list(set(desired_actions) & set(actions))

		if len(possible_actions) == 0: return actions
		else: return possible_actions

	# Removes possible actions due to the boundary of the map
	def checkMapBoundaries(self, actions):
		# Check map boundaries in x axis
		if self._agent_x == 0:
			actions.remove('L')
		elif self._agent_x == self._grid_width - 1:
			actions.remove('R')

		# Check map boundaries in y axis
		if self._agent_y == 0:
			actions.remove('F')
		elif self._agent_y == self._grid_height - 1:
			actions.remove('B')

		return actions

	def checkVisitedLocations(self, actions):
		# List to be returned of possible actions
		possible_actions = []

		# Current agent position
		x = self._agent_x
		y = self._agent_y

		#Iterate over the supplied possible actions
		for action in actions:
			if action == 'F':
				if not self._map[y-1,x]:
					possible_actions.append('F')
			elif action == 'B':
				if not self._map[y+1,x]:
					possible_actions.append('B')
			elif action == 'L':
				if not self._map[y,x-1]:
					possible_actions.append('L')
			elif action == 'R':
				if not self._map[y,x+1]:
					possible_actions.append('R')
			else:
				print "Action: {} not recognised!".format(action)

		return possible_actions

	def performAction(self, action):
		# Action performed
		# print "Action performed = {}".format(action)
		# print self._map

		# Make the move
		if action == 'F': 	self._agent_y -= self._move_dist
		elif action == 'B': self._agent_y += self._move_dist
		elif action == 'L': self._agent_x -= self._move_dist
		elif action == 'R': self._agent_x += self._move_dist
		else: print "Action: {} not recognised!".format(action)

		# Check whether the agent has now visited a target
		ret = self.checkAgentTargetMatch()

		# Record history of map visitation
		self._map[self._agent_y, self._agent_x] = 1

		return ret

	# Assign a point if a target has been visited for the first time
	def checkAgentTargetMatch(self):
		a_x = self._agent_x
		a_y = self._agent_y

		for target in self._targets:
			t_x = target[0]
			t_y = target[1]

			if t_x == a_x and t_y == a_y:
				if not self._map[a_y, a_x]:
					return 1

		return 0

	# Given a position x,y return neighbouring values within a given radius
	def determineCellNeighbours(self, x, y, radius):
		# Double the radius
		d_rad = radius * 2

		# Add padding to coordinates
		x += radius
		y += radius

		# Pad a temporary map with ones
		padded_map = np.ones((self._grid_width+d_rad, self._grid_height+d_rad))

		# Insert visitation map into border padded map
		padded_map[		radius:self._grid_width+radius,
						radius:self._grid_height+radius	] = self._map[:,:]

		# Determine neighbouring cell bounds for radius
		y_low = y - radius
		y_high = y + radius
		x_low = x - radius
		x_high = x + radius

		# Get neighbouring elements within radius (includes x,y-th element)
		sub = padded_map[y_low:y_high+1, x_low:x_high+1]

		# Get indices of elements that are unvisited (0)
		indices = np.where(sub == 0)

		# Action to carry out
		action = None

		# Check whether some 0 elements were found
		if indices[0].size > 0 and indices[1].size > 0:
			# Agent position in subview
			a_x = np.floor((d_rad+1)/2)
			a_y = np.floor((d_rad+1)/2)

			# 0 element position in subview
			z_x = indices[0][0]
			z_y = indices[1][0]

			# Find the best action for the angle between them
			action = self.findActionForAngle((a_x, a_y), (z_x, z_y))

		print sub
		print indices
		cv2.waitKey(0)

		return action

	def findUnvisitedDirection(self):
		# Cell search radius for unvisited cells
		radius = 1

		action = None

		while action is None:
			action = self.determineCellNeighbours(self._agent_x, self._agent_y, radius)

			radius += 1

		return action

	# Given the position of a target, find the angle between the agent position and
	# the target and choose the best possible action towards navigating towards that
	# target object
	def findActionForAngle(self, a, b):
		# Get relative position
		rel_x = a[0] - b[0]
		rel_y = a[1] - b[1]

		# Compute angle
		angle = math.atan2(rel_x, rel_y)

		# print "Angle = {} for point ({},{})".format(math.degrees(angle), rel_x, rel_y)

		# Move forward
		if angle < math.pi/4 and angle > -math.pi/4:
			#print "Moving forward"
			action = 'F'
		# Move left
		elif angle >= math.pi/4 and angle < 3*math.pi/4:
			#print "Moving left"
			action = 'L'
		# Move right
		elif angle <= math.pi/4 and angle > -3*math.pi/4:
			#print "Moving right"
			action = 'R'
		# Move backward
		elif angle >= 3*math.pi/4 or angle <= -3*math.pi/4:
			#print "Moving backward"
			action = 'B'

		return action

	# Returns the Euclidean distance between input coordinates a, b in tuple form (x, y)
	def findDistanceBetweenPoints(self, a, b):
		return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

	# Returns the coordinates of the closest target to the current agent position that
	# hasn't already been visited
	def findClosestTarget(self):
		# Get the current agent position
		a_x = self._agent_x
		a_y = self._agent_y

		best_dist = float("inf")
		best_coords = (-1,-1)

		# Iterate over each target
		for target in self._targets:
			# Check that we haven't already visited this target
			if not self._map[target[1],target[0]]:
				# Find the distance
				distance = self.findDistanceBetweenPoints((a_x, a_y), target)

				# Is the current distance better
				if distance < best_dist:
					best_dist = distance
					best_coords = target

		return best_coords

	def actionToClassVector(self, action):
		vec = np.zeros(len(self._actions))

		if action == 'F': vec[0] = 1
		elif action == 'B': vec[1] = 1
		elif action == 'L': vec[2] = 1
		elif action == 'R': vec[3] = 1
		else:
			print "Action not recognised, oops.."

		return vec

	def classVectorToAction(self, class_vec):
		action = ''

		if class_vec[0]: action = 'F'
		elif class_vec[1]: action = 'B'
		elif class_vec[2]: action = 'L'
		elif class_vec[3]: action = 'R'
		else:
			print "Action not recognised, oops.."

		return action

	def begin(self):
		# Render the updated view
		self._render_img, self._agent_subview = self.render()

		# Display if we're supposed to
		if self._visualise:
			# Display the image
			cv2.imshow(self._window_name, self._render_img)
			#cv2.waitKey(0)

		# All possible agent actions
		all_actions = self._actions

		# Number of targets the agent has visited
		num_visited = 0

		# Action that the agent ends up choosing to perform
		chosen_action = None

		# Loop until we've visited all the target objects
		while num_visited != self._num_targets:
			# Agent can view the entire map and the memory map to make a perfect decision
			# about which target to visit
			# It simply finds the closest target and moves towards that, repeating until
			# all targets have been visited
			if self._agent_has_global_view:
				# Find the coordinates of the closest target to the current agent position
				closest_target = self.findClosestTarget()

				# Agent's current position
				agent_pos = (self._agent_x, self._agent_y)

				# print "Current agent position: {}".format(agent_pos)
				# print "Current closest target at: {}".format(closest_target)

				# Find the best action for the closest target
				chosen_action = self.findActionForAngle(agent_pos, closest_target)
			# Assume the agent only can see the memory map and a partial view of the world
			# PODMP
			else:
				# Remove impossible actions imposed by map boundary
				possible_actions = self.checkMapBoundaries(list(all_actions))

				# Create a seperate list of possible actions discluding visited locations
				visit_actions = self.checkVisitedLocations(possible_actions)

				# Ideally navigate towards target (if we can see one)
				desired_actions = self.checkVisibility(visit_actions)

				# There isn't anywhere to go, engage un-stucking mode
				if not len(desired_actions):
					#print "Oops, I'm stuck."

					# Try to move towards an unvisited location
					chosen_action = self.findUnvisitedDirection()
				else:
					# Choose a random possible action (for the time being)
					chosen_action = desired_actions[random.randint(0, len(desired_actions)-1)]

			# Print out score
			#print "I've visted {} targets".format(num_visited)

			# Save the subimage, memory map and action (class)
			if self._save_output:
				# Make a copy of the map
				map_copy = self._map.copy()

				# Set the agent position to 10 in the map
				map_copy[self._agent_y, self._agent_x] = 10

				action_vector = self.actionToClassVector(chosen_action)

				row = [self._agent_subview, map_copy, action_vector]
				self._training_output.append(row)

			# Make the move
			num_visited += self.performAction(chosen_action)

			# Render the updated view
			self._render_img, self._agent_subview = self.render()

			# Display if we're supposed to
			if self._visualise:
				# Display the image
				cv2.imshow(self._window_name, self._render_img)
				cv2.imshow(self._window_name_agent, self._agent_subview)
				cv2.waitKey(0)

	# Reset the map (agent position, target positions, memory, etc.)
	def reset(self):
		# Reset the agent randomly or not
		if self._random_agent_pos:
			self._agent_x = random.randint(0, self._grid_width-1)
			self._agent_y = random.randint(0, self._grid_height-1)
		else:
			self._agent_x = 0
			self._agent_y = 0

		# Reset the map
		# Grid-visit map of environment (Agent has access to this)
		# Map is binary (cells are either 1: visited, 0: not visited)
		self._map = np.zeros((self._grid_width, self._grid_height))

		# Record the current agent position in the visitation map
		self._map[self._agent_y, self._agent_x] = 1

		# Randomly initialise positions of targets, ensure they don't
		# take up the positions of other targets or the agent
		self._targets = self.initTargets()

	# Do a given number of episodes
	def startXEpisodes(self, num_episodes):
		for i in range(num_episodes):
			self.reset()
			self.begin()

			print "{}/{}, {}% complete".format(i+1, num_episodes, (float(i+1)/num_episodes)*100)

		# Save the output if we're supposed to
		if self._save_output:
			with open(self._data_dir, 'wb') as fout:
				pickle.dump(self._training_output, fout)

# Entry method
if __name__ == '__main__':
	### Generating training data
	# fm = FieldMap(visualise=False, agent_global_view=True, save=True)
	# fm.startXEpisodes(20000)

	### Training model on synthesised data
	# fm = FieldMap(visualise=False, agent_global_view=True, save=True)
	# model = dnn_model(fm)
	# model.trainModel()

	### Testing trained model on real example/problem
	fm = FieldMap(visualise=True)
	model = dnn_model(fm)
	model.testModelOnRealExample()
