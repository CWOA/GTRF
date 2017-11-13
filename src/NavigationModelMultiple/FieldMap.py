#!/usr/bin/env python

import cv2
import DNN
import Object
import random
import numpy as np
from Utility import *
import Visualisation
import VisitationMap
from tqdm import tqdm
import Constants as const

class FieldMap:
	# Class constructor
	def __init__(		self, 
						visualise=False,
						use_simulator=True,
						random_agent_pos=True,
						training_model=False,
						save=False						):
		"""
		Class arguments from init
		"""

		# Bool to decide whether to actually visualise
		self._visualise = visualise

		# Position of agent should be generated randomly
		self._random_agent_pos = random_agent_pos

		# Whether or not we should be saving output to file
		self._save_output = save

		# Whether or not we should use ROS/gazebo simulator
		self._use_simulator = use_simulator

		# If we're just training the DNN
		self._training_model = training_model

		"""
		Class attributes
		"""

		# Don't initialise the Visualiser (and ROS node) if we're just training
		if not self._training_model:
			# Class in charge of handling agent/targets
			self._object_handler = Object.ObjectHandler()

			# Class in charge of visitation map
			self._map_handler = VisitationMap.MapHandler()

			# Class in charge of visualisation (for both model input and our viewing benefit)
			self._visualiser = Visualisation.Visualiser(self._use_simulator)

			# Initialise the agent loop detection module
			self._loop_detector = LoopDetector()

			# Training data list to pickle upon completion (if we're supposed to be
			# saving output)
			self._training_output = []

		# Deep Neural Network class for model prediction, training, etc.
		self._dnn = DNN.DNNModel(self._use_simulator)

		"""
		Class setup
		"""

		# Don't initialise this class if we're just training
		if not self._training_model:
			# Initialise all the necessary elements
			self.reset()

	# Reset the map (agent position, target positions, memory, etc.)
	def reset(self):
		# Reset objects (agent, target), returns generated agent/target positions
		states = self._object_handler.reset()
		a_x = states[0][0]
		a_y = states[0][1]
		target_poses = states[1]

		# If we're using the gazebo simulator, move the agent/targets to generated positions
		if self._use_simulator:
			self._visualiser.resetAgentTargets(a_x, a_y, target_poses)

		# Reset the visit map
		self._map_handler.reset(a_x, a_y)

		# Reset loop detection
		self._loop_detector.reset()

	# Perform a given action
	def performAction(self, action):
		# Get the agent's current position
		old_x, old_y = self._object_handler.getAgentPos()

		# Make a copy
		req_x = old_x
		req_y = old_y

		# Make the move
		if action == 'F': 	req_y -= const.MOVE_DIST
		elif action == 'B': req_y += const.MOVE_DIST
		elif action == 'L': req_x -= const.MOVE_DIST
		elif action == 'R': req_x += const.MOVE_DIST
		else: Utility.die("Action: {} not recognised!".format(action))

		# Requested position is in bounds
		if self._map_handler.checkPositionInBounds(req_x, req_y):
			# Set the new agent position
			self._object_handler.setAgentPos(req_x, req_y)
		# Agent tried to move out of bounds, select a random valid action instead
		else:
			# Find possible actions from all actions given the map boundaries
			possible_actions = self._map_handler.possibleActionsForPosition(old_x, old_y)

			# Randomly select an action
			rand_idx = random.randint(0, len(possible_actions)-1)
			choice = possible_actions[rand_idx]

			# Recurse to perform selected action
			return self.performAction(choice)

		# Update the map, function returns whether this new position
		# has been visited before
		is_new_location = self._map_handler.update(req_x, req_y)

		return is_new_location

	# Retrieves the current agent position, list of target positions and visitation map
	def retrieveStates(self):
		# Get the agent position
		pos = self._object_handler.getAgentPos()

		# Get all the target positions
		targets_pos = self._object_handler.getTargetPositions()

		# Get the visit map
		visit_map = self._map_handler.getMap()

		return (pos, targets_pos, visit_map)

	# Given the current agent subview and visit map, use the trained DNN model to predict
	# the best possible action in this circumstance
	def predictBestAction(self, subview, visit_map):
		# Predict using DNN, returns probabilty score list for each class
		probability_vec = self._dnn.testModelSingle(subview, visit_map)

		# Find index of max value
		max_idx = np.argmax(probability_vec)

		# Create a new probability vector with the max index = 1
		choice_vec = np.zeros(len(const.ACTIONS))
		choice_vec[max_idx] = 1

		# Convert to action
		return Utility.classVectorToAction(choice_vec)

	def beginInstance(self, testing, wait_amount=0):
		# Render the initial game state
		_, subview = self._visualiser.update(self.retrieveStates())

		# Number of moves the agent has made
		num_moves = 0

		# Display if we're supposed to
		if self._visualise: self._visualiser.display(wait_amount)

		# Indicator of whether the agent is stuck in an infinite loop
		if testing: agent_stuck = False

		# Loop until we've visited all the target objects
		while not self._object_handler.allTargetsVisited():
			# Use the DNN model to make action decisions
			if testing:
				# Get the map
				visit_map = self._map_handler.getMap()

				# Use DNN model to predict correct action
				chosen_action = self.predictBestAction(subview, visit_map)

				# Add the suggested action and check history, check if the agent is
				# stuck in a loop, act accordingly
				if not agent_stuck and self._loop_detector.addCheckAction(chosen_action):
					agent_stuck = True
					# print "Agent stuck, entering unstucking mode"

				# Agent is stuck, move towards nearest unvisited location
				if agent_stuck:
					a_x, a_y = self._object_handler.getAgentPos()
					chosen_action = self._map_handler.findUnvisitedDirection(a_x, a_y)

			# We're just producing training instances
			else:
				# Find the coordinates of the closest target to the current agent position
				agent_pos, closest_target = self._object_handler.findClosestTarget()

				# Find the best action for the closest target
				chosen_action = Utility.bestActionForAngle(	agent_pos[0],
															agent_pos[1],
															closest_target[0],
															closest_target[1]	)

				# Save the subimage, memory map and action (class)
				if self._save_output:
					self.recordData(	subview, 
										self._map_handler.getMap(),
										Utility.actionToClassVector(chosen_action)	)

			# Make the move
			is_new_location = self.performAction(chosen_action)

			# Check whether the agent is still stuck
			if testing and agent_stuck and is_new_location:
				# Delete elements in the loop detector
				self._loop_detector.reset()

				# Indicate that the agent is no longer stuck
				agent_stuck = False

				# print "Agent is no longer stuck!"

			# Increment the move counter
			num_moves += 1

			# Render the updated views (for input into the subsequent iteration)
			_, subview = self._visualiser.update(self.retrieveStates())

			# Display if we're supposed to
			if self._visualise: self._visualiser.display(wait_amount)

		return num_moves

	# For some timestep, append data to the big list
	def recordData(self, subview, visit_map, action_vector):
		# Create list of objects for this timestep
		row = [subview, visit_map, action_vector]

		# Add it to the list
		self._training_output.append(row)

	# Save output data to file
	def saveDataToFile(self):
		# Use HDF5 py
		if const.USE_HDF5:
			import h5py

			print "Saving data using h5py"

			# Open the dataset file (may overwrite an existing file!!)
			dataset = h5py.File(Utility.getHDF5DataDir(), 'w')

			# The number of training instances generated
			num_instances = len(self._training_output)

			# The number of possible action classes
			num_classes = len(const.ACTIONS)

			# Image dimensions
			img_width = const.IMG_DOWNSAMPLED_WIDTH
			img_height = const.IMG_DOWNSAMPLED_HEIGHT
			channels = const.NUM_CHANNELS

			# Create three datasets within the file with the correct shapes:
			# X0: agent visual subview
			# X1: visitation map
			# Y: corresponding ground truth action vector in form [0, 1, 0, 0]
			dataset.create_dataset('X0', (num_instances, img_width, img_height, channels))
			dataset.create_dataset('X1', (num_instances, const.MAP_WIDTH, const.MAP_WIDTH))
			dataset.create_dataset('Y', (num_instances, num_classes))

			# Actually add instances to the respective datasets
			for i in range(len(self._training_output)):
				dataset['X0'][i] = self._training_output[i][0]
				dataset['X1'][i] = self._training_output[i][1]
				dataset['Y'][i] = self._training_output[i][2]

			# Finish up
			dataset.close()
		# Use pickle
		else:
			import pickle

			print "Pickling/saving data, this may take some time..."

			# Save it out
			with open(Utility.getPickleDataDir(), 'wb') as fout:
					pickle.dump(self._training_output, fout)

		print "Finished saving data!"

	# Do a given number of episodes
	def startTrainingEpisodes(self, num_episodes):
		# Initialise progress bar (TQDM) object
		pbar = tqdm(total=num_episodes)

		for i in range(num_episodes):
			self.beginInstance(False, wait_amount=const.WAIT_AMOUNT)
			self.reset()

			pbar.update()

			# print "{}/{}, {}% complete".format(i+1, num_episodes, (float(i+1)/num_episodes)*100)

		pbar.close()

		# Save the output if we're supposed to
		if self._save_output: self.saveDataToFile()

	# Do a given number of testing episodes
	def startTestingEpisodes(self, num_episodes):
		# Load the DNN model from file
		self._dnn.loadSaveModel()

		# The exhaustive number of moves to visit every location with this map size
		upper_num_moves = const.MAP_WIDTH * const.MAP_HEIGHT

		# Number of testing instances with the number of moves below the exhaustive amount
		num_under = 0

		print "Beginning testing on generated examples"

		# Do some testing episodes
		for i in range(num_episodes):
			num_moves = self.beginInstance(True, wait_amount=const.WAIT_AMOUNT)
			self.reset()

			if num_moves < upper_num_moves: num_under += 1

			print "Solving example {}/{} took {} moves".format(i+1, num_episodes, num_moves)

		# Print some more stats
		percent_correct = float(num_under/num_examples) * 100
		print "{}/{} under {} moves, or {}% success".format(	num_under,
																num_examples,
																upper_num_moves,
																percent_correct		)

	# Signifies DNN class to train model on data at a given directory
	def trainModel(self):
		self._dnn.trainModel()

# Entry method/unit testing
if __name__ == '__main__':
	pass	
