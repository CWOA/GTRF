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
from Algorithms.Algorithm import Algorithm

"""
This class forms the principal managerial component of this framework and directs episode
generation, execution or model training
"""

class FieldMap:
	# Class constructor
	def __init__(		self, 
						visualise=False,
						use_simulator=True,
						random_agent_pos=True,
						training_model=False,
						save=False,
						second_solver=False		):
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

		# Don't initialise the Visualiser (and ROS node) if we're just training the DNN
		if not self._training_model:
			# Algorithm class for selecting agent actions based on the environment state
			# You can override the algorithm method here
			self._algorithm = Algorithm(const.ALGORITHM_METHOD, self._use_simulator)

			# Class in charge of handling agent/targets
			self._object_handler = Object.ObjectHandler(second_solver=second_solver)

			# Class in charge of visitation map
			self._map_handler = VisitationMap.MapHandler()

			# Class in charge of visualisation (for both model input and our viewing benefit)
			self._visualiser = Visualisation.Visualiser(self._use_simulator)

			# Training data list to save upon completion (if we're even supposed to be
			# saving output at all)
			self._training_output = []

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

		# Reset the visitation map
		self._map_handler.reset(a_x, a_y)

		# Reset the algorithm
		self._algorithm.reset()

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
		if Utility.checkPositionInBounds(req_x, req_y):
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

	# Begin this episode whether we're generating training data, testing, etc.
	def beginEpisode(self, testing, wait_amount=0):
		# Render the initial episode state
		_, subview = self._visualiser.update(self.retrieveStates())

		# Number of moves the agent has made
		num_moves = 0

		# Display if we're supposed to
		if self._visualise: self._visualiser.display(wait_amount)

		# Indicate to the solver to solve this episode/instance, returns the length
		# of the generated solution using the selected solver method
		# This is typically used for extracting the global optimum solution to a 
		# particular episode configuration
		sol_length = self._object_handler.solveEpisode()

		# Loop until we've visited all the target objects
		while not self._object_handler.allTargetsVisited():
			# Use the selected Algorithm to choose actions based on the given input
			if testing:
				# Get the current state of the occupancy map
				occupancy_map = self._map_handler.getMap()

				# Use Algorithm to choose an action
				chosen_action = self._algorithm.iterate(subview, occupancy_map)

			# We're just producing training instances
			else:
				# Get the next selected action from the solver
				chosen_action = self._object_handler.nextSolverAction()

				# Save the subimage, memory map and action (class)
				if self._save_output:
					self.recordData(	subview, 
										np.copy(self._map_handler.getMap()),
										Utility.actionToClassVector(chosen_action)	)

			# Make the move
			_ = self.performAction(chosen_action)

			# Iterate the object handler
			self._object_handler.iterate()

			# Increment the move counter
			num_moves += 1

			# Render the updated views (for input into the subsequent iteration)
			_, subview = self._visualiser.update(self.retrieveStates())

			# Display if we're supposed to
			if self._visualise: self._visualiser.display(wait_amount)

		# Retrieve the number of loops detected. 0 for algorithms that don't use it
		num_loops = self._algorithm.getNumLoops()

		# Return the number of moves taken by the model and the solution
		# Also return the number of times loop detection is found
		return num_moves, sol_length, num_loops

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

			# file_str = Utility.getHDF5DataDir()
			file_str = "{}/gaussian_SEQUENCE.h5".format(Utility.getICIPDataDir())

			# Open the dataset file (may overwrite an existing file!!)
			dataset = h5py.File(file_str, 'w')

			# The number of training instances generated
			num_instances = len(self._training_output)

			# The number of possible action classes
			num_classes = len(const.ACTIONS)

			# Image dimensions
			if self._use_simulator:
				img_width = const.IMG_DOWNSAMPLED_WIDTH
				img_height = const.IMG_DOWNSAMPLED_HEIGHT
				channels = const.NUM_CHANNELS
			else:
				img_width = const.GRID_PIXELS * 3
				img_height = const.GRID_PIXELS * 3
				channels = const.NUM_CHANNELS

			# Create three datasets within the file with the correct shapes:
			# X0: agent visual subview
			# X1: visitation map
			# Y: corresponding ground truth action vector in form [0, 1, 0, 0]
			dataset.create_dataset('X0', (num_instances, img_width, img_height, channels))
			dataset.create_dataset('X1', (num_instances, const.MAP_WIDTH, const.MAP_HEIGHT))
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
			self.reset()
			self.beginEpisode(False, wait_amount=const.WAIT_AMOUNT)

			pbar.update()

		pbar.close()

		# Save the output if we're supposed to
		if self._save_output: self.saveDataToFile()

	# Do a given number of testing episodes
	def startTestingEpisodes(self, num_episodes):
		print "Beginning testing on generated examples"

		# Place to store testing data to in numpy form
		base = Utility.getICIPDataDir()

		# Numpy array for testing data, consists of:
		# 0: number of moves required by the model
		# 1: number of moves required by employed solver (closest, target ordering)
		# 2: number of times loop detection is triggered
		test_data = np.zeros((num_episodes, 3))

		# Initialise progress bar (TQDM) object
		pbar = tqdm(total=num_episodes)

		# Do some testing episodes
		for i in range(num_episodes):
			# Reset (generate a new episode)
			self.reset()

			# Go ahead and solve this instance using model & solver for comparison
			num_moves, sol_length, num_loops = self.beginEpisode(True, wait_amount=const.WAIT_AMOUNT)

			# Store statistics to numpy array
			test_data[i,0] = num_moves
			test_data[i,1] = sol_length
			test_data[i,2] = num_loops

			# Update progress bar
			pbar.update()

		# Close up
		pbar.close()

		# Save data to file
		np.save("{}/test_data_GAUS_SEQ".format(base), test_data)

	# Compare solver performance over a number of testing episodes
	def compareSolvers(self, num_episodes):
		print "Beginning comparing solvers"

		# Place to store testing data to in numpy form
		base = Utility.getICIPDataDir()

		# Numpy array for testing data, consists of:
		# 0: number of moves required by the model
		# 1: number of moves required by employed solver (closest, target ordering)
		# 2: number of times loop detection is triggered
		test_data = np.zeros((num_episodes, 3))

		# Initialise progress bar (TQDM) object
		pbar = tqdm(total=num_episodes) 

		# Do some comparisons
		for i in range(num_episodes):
			# Reset (generate a new episode)
			self.reset()

			# Get solution lengths (NS: naive solver, GO: globally-optimal)
			NS_length = self._object_handler.secondSolveEpisode()
			GO_length = self._object_handler.solveEpisode()

			# Store statistics to numpy array
			test_data[i,0] = NS_length
			test_data[i,1] = GO_length
			test_data[i,2] = 0	# There's no loop detection here

			# Update progress bar
			pbar.update()

		# Close up
		pbar.close()

		# Save data to file
		np.save("{}/test_data_NAIVE".format(base), test_data)

	# Generate solutions to randomly-generated episodes using trained model and output
	# visualisation of agent path to file if the difference between the generated
	# and globally-optimal solution are within a defined range
	def generateVisualisations(self, dif_range, num_images=25):
		print "Beginning generating visualisations to episodes"
		print "Trying to generate {} images in range {}".format(num_images, dif_range)

		# Load the DNN model from file
		self._dnn.loadSaveModel()

		# Place to store images to
		base = os.path.join(Utility.getICIPFigureDir(), "raw_instances")

		# Initialise progress bar (TQDM) object
		pbar = tqdm(total=num_images)

		# Image scale factor
		sf = 10

		# File counter
		i = 0

		# Loop until we've created enough images
		while i < num_images:
			# Reset (generate a new episode)
			self.reset()

			# Solve this instance
			moves, sol_length, _ = self.beginEpisode(True, wait_amount=1)

			# Get the difference in solution lengths
			dif = moves - sol_length

			# Check the difference is in range
			if dif >= dif_range[0] and dif <= dif_range[1]:
				# Get the final image for this instance
				img = self._visualiser._render_img

				# Resize image to something reasonable
				img = cv2.resize(img, None, fx=sf, fy=sf, interpolation=cv2.INTER_NEAREST)

				# Construct save string (file to save to)
				file_path = os.path.join(base, "{}.jpg".format(i))

				# Save it
				cv2.imwrite(file_path, img)

				# Update progress bar
				pbar.update()

				# Increment the counter
				i += 1

		# Close up
		pbar.close()


	# Signifies DNN class to train model on data at a given directory
	def trainModel(self):
		self._dnn.trainModel()

# Entry method/unit testing
if __name__ == '__main__':
	# Generate visualisations of runs for ICIP/thesis
	fm = FieldMap(visualise=True, use_simulator=False)
	fm.generateVisualisations((40, 50), num_images=25)
