#!/usr/bin/env python

import DNN
import Object
import random
import numpy as np
from Utility import *
import Visualisation
import VisitationMap
import Constants as const

class FieldMap:
	# Class constructor
	def __init__(		self, 
						visualise=False,
						random_agent_pos=True, 
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

		"""
		Class attributes
		"""

		# Class in charge of handling agent/targets
		self._object_handler = Object.ObjectHandler()

		# Class in charge of visitation map
		self._map_handler = VisitationMap.MapHandler()

		# Class in charge of visualisation (for both model input and our benefit)
		self._visualiser = Visualisation.Visualiser()

		# Initialise the agent loop detection module
		self._loop_detector = LoopDetector()

		# Deep Neural Network class for model prediction, training, etc.
		self._dnn = DNN.DNNModel()

		# Training data list to pickle upon completion (if we're supposed to be
		# saving output)
		self._training_output = []

		"""
		Class setup
		"""

		# Initialise all the necessary elements
		self.reset()

	# Reset the map (agent position, target positions, memory, etc.)
	def reset(self):
		# Reset objects (agent, target), returns generated agent position
		a_x, a_y = self._object_handler.reset()

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
		complete_img, subview = self._visualiser.update(self.retrieveStates())

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

				# Agent is stuck, move towards nearest unvisited location
				if agent_stuck:
					a_x, a_y = self._object_handler.getAgentPos()
					chosen_action = self._map_handler.findUnvisitedDirection(a_x, a_y)

			# We're just producing training instances
			else:
				# Find the coordinates of the closest target to the current agent position
				closest_target = self._object_handler.findClosestTarget()

				# Fetch agent's current position
				agent_pos = self._object_handler.getAgentPos()

				# Find the best action for the closest target
				chosen_action = self.findActionForAngle(agent_pos, closest_target)

				# Save the subimage, memory map and action (class)
				if self._save_output:
					self.recordData(	subview, 
										self._map_handler.getMap(),
										Utility.actionToClassVector(chosen_action)	)

			# Make the move
			is_new_location = self.performAction(chosen_action)

			# Check whether the agent is still stuck
			if agent_stuck and testing and is_new_location: agent_stuck = False

			# Increment the move counter
			num_moves += 1

			# Render the updated view
			complete_img, subview = self._visualiser.update(self.retrieveStates())

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
		print "Pickling/saving data, this may take some time..."

		# Save it out
		with open(Utility.getDataDir(), 'wb') as fout:
				pickle.dump(self._training_output, fout)

		print "Finished saving data!"

	# Do a given number of episodes
	def startTrainingEpisodes(self, num_episodes):
		for i in range(num_episodes):
			self.beginInstance(False)
			self.reset()

			print "{}/{}, {}% complete".format(i+1, num_episodes, (float(i+1)/num_episodes)*100)

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

# Entry method/unit testing
if __name__ == '__main__':
	pass	
