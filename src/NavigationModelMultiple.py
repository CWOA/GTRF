#!/usr/bin/env python

import os
import cv2
import random
import tflearn
import numpy as np

# Navigation model class for MULTIPLE STATIC objectives
class NavigationModelMultiple:
	# Class constructor
	def __init__(self, load_model=False):
		# Base directory for TFLearn attributes
		self._base_dir = "/home/will/catkin_ws/src/uav_id/tflearn"

		# Directories
		self._checkpoint_dir = os.path.join(self._base_dir, "checkpoints/")
		self._tensorboard_dir = os.path.join(self._base_dir, "tensorboard/")

		# Number of possible classes (do nothing, forward, backward, left, right)
		# Unsure whether to include "no nothing" class
		self._num_classes = 5

		# Dimensions to resize images to (training or testing)
		self._resize_width = 200
		self._resize_height = 200

		# Number of image channels
		self._num_channels = 3

		# How verbose tensorboard is (0: fastest, 3: most detail)
		self._tensorboard_verbose = 0

		# Network declaration
		self._network = self.defineNetwork(		self._num_classes,
												self._resize_width,
												self._resize_height,
												self._num_channels		)

		# Model declaration
		self._model = tflearn.DNN(		self._network,
										checkpoint_path=self._checkpoint_dir,
										max_checkpoints=3,
										tensorboard_verbose=self._tensorboard_verbose,
										tensorboard_dir=self._tensorboard_dir			)

		# Should we load model weights (primarily for testing)
		if load_model:
			self.loadModel()

	# Neural network architecture definition
	def defineNetwork(self, num_classes, width, height, channels):
		pass


	def generateTrainingInstance(self):

		pass

		# finished = False

		# while not finished:
		# 	# List of all possible actions
		# 	actions = ['']

		# 	# See whether a cow is visible, if so, in which direction
		# 	direction = isCowVisible()

		# 	# A cow is currently visible
		# 	if direction > 0:

		# 	# A cow is no currently visible
		# 	else:

class FieldMap:
	# Class constructor
	def __init__(self, visualise=False):
		### Class attributes

		# Dimensions of grid
		self._grid_width = 10
		self._grid_height = 10

		# Current agent position (origin is top, left)
		self._agent_x = 0
		self._agent_y = 0

		# Unit to move agent by each movement
		self._move_dist = 1

		# Number of target objects (number of cows)
		self._num_targets = 5

		# Randomly initialise positions of targets
		self._targets = self.initTargets()

		# Possible agent actions
		self._actions = ['F', 'B', 'L', 'R']

		# Grid-visit map of environment (Agent has access to this)
		# Map is binary (cells are either 1: visited, 0: not visited)
		self._map = np.zeros((self._grid_width, self._grid_height))

		### Display attributes

		# Bool to decide whether to actually visualise
		self._visualise = visualise

		# Number of pixels a grid takes up
		self._grid_pixels = 100

		# Dimensions of display/visualisation grid
		self._disp_width = self._grid_pixels * self._grid_width
		self._disp_height = self._grid_pixels * self._grid_height

		# Window name
		self._window_name = "Visualisation grid"

		# Colours (BGR)
		self._background_colour = (42,42,23)
		self._visited_colour = (181,161,62)
		self._agent_colour = (89,174,110)
		self._target_colour = (64,30,162)

	def visualise(self):
		# Create image to render to
		img = np.zeros((self._disp_height, self._disp_width, 3), np.uint8)

		# Set image to background colour
		img[:] = self._background_colour

		# Render target locations
		for target in self._targets:
			img = self.renderGridPosition(target[0], target[1], img, self._target_colour)

		# Render visited locations
		for x in range(self._grid_width):
			for y in range(self._grid_height):
				# Have we been to this coordinate before?
				if self._map[x,y]:
					# Render this square as have being visited
					img = self.renderGridPosition(x, y, img, self._visited_colour)

		# Render current agent position
		img = self.renderGridPosition(		self._agent_x, 
											self._agent_y, 
											img, 
											self._agent_colour		)

		# Display the image
		cv2.imshow(self._window_name, img)
		cv2.waitKey(0)

	def renderGridPosition(self, x, y, img, colour):
		img[x*self._grid_pixels:(x+1)*self._grid_pixels,
			y*self._grid_pixels:(y+1)*self._grid_pixels,:] = colour

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
				targets.append((x_pos, y_pos))
				num_generated += 1

		return targets

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
				if not self._map[x,y-1]:
					possible_actions.append('F')
			elif action == 'B':
				if not self._map[x,y+1]:
					possible_actions.append('B')
			elif action == 'L':
				if not self._map[x-1,y]:
					possible_actions.append('L')
			elif action == 'R':
				if not self._map[x+1,y]:
					possible_actions.append('R')
			else:
				print "Action: {} not recognised!".format(action)

		return possible_actions

	def performAction(self, action):
		# Record history of map visitation
		self._map[self._agent_x, self._agent_y] = 1

		# Make the move
		if action == 'F': 	self._agent_y -= self._move_dist
		elif action == 'B': self._agent_y += self._move_dist
		elif action == 'L': self._agent_x -= self._move_dist
		elif action == 'R': self._agent_x += self._move_dist
		else: print "Action: {} not recognised!".format(action)

	# Reset the map (agent position, target positions, memory, etc.)
	def reset(self):
		pass

	def begin(self):
		# Need to decide what termination criterion actually are
		finished = False

		# All possible agent actions
		all_actions = ['F', 'B', 'L', 'R']

		while not finished:
			# Render out if we're supposed to
			if self._visualise:
				self.visualise()

			# Remove impossible actions imposed by map boundary
			possible_actions = self.checkMapBoundaries(list(all_actions))

			# Create a seperate list of possible actions discluding visited locations
			visit_actions = self.checkVisitedLocations(possible_actions)

			# There isn't anywhere to go!
			if not len(visit_actions):
				print "I'm stuck here!"
			else:
				# Choose a random possible action (for the time being)
				choice = visit_actions[random.randint(0, len(visit_actions)-1)]

				# Make the move
				self.performAction(choice)

			# Termination critertion for the moment is that each cell has been visited
			if np.sum(self._map) == self._grid_width * self._grid_height:
				finished = True

# Entry method
if __name__ == '__main__':
	fm = FieldMap(visualise=True)
	fm.begin()
