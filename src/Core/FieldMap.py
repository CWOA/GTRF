#!/usr/bin/env python

# Core libraries
import sys
sys.path.append('../')
import cv2
import h5py
import random
import numpy as np
from tqdm import tqdm

# My classes
import Object
from Utilities.Utility import Utility
from Utilities.VideoWriter import VideoWriter
from Utilities.ResultsHelper import ResultsHelper
import Visualisation
import VisitationMap
import Constants as const
from Algorithms.Algorithm import Algorithm

"""
This class forms the principal managerial component of this framework and directs episode
generation, execution or model training
"""

class FieldMap:
	# Class constructor
	def __init__(		self,
						generating,
						exp_name,
						visualise=False,
						use_simulator=False,
						random_agent_pos=True,
						save=False,
						second_solver=False,
						model_path=None,
						save_video=False,
						dist_method=const.OBJECT_DIST_METHOD,
						mark_visitation=const.MARK_PAST_VISITATION		):
		"""
		Class arguments from init
		"""

		# Whether we're just generating training examples 
		self._generating = generating

		# Bool to decide whether to actually visualise
		self._visualise = visualise

		# Position of agent should be generated randomly
		self._random_agent_pos = random_agent_pos

		# Whether or not we should be saving output to file
		self._save_output = save

		# Whether or not we should use ROS/gazebo simulator
		self._use_simulator = use_simulator

		# Whether we should save each episode as a video
		self._save_video = save_video

		# Name of this experiment
		self._exp_name = exp_name

		"""
		Class attributes
		"""

		# If we should generate video
		if self._save_video:
			self._vw_GO_I = VideoWriter(self._exp_name, Utility.getVideoDir())
			self._vw_GO_M = VideoWriter(self._exp_name, Utility.getVideoDir())

		if not self._generating:
			# Algorithm class for selecting agent actions based on the environment state
			# You can override the algorithm method here
			self._algorithm = Algorithm(const.ALGORITHM_METHOD, self._use_simulator, model_path)
			
			# If we should generate video
			if self._save_video:
				# Initialise three video writers for our solution, agent view, agent occupancy map
				self._vw_OURS = VideoWriter(self._exp_name, Utility.getVideoDir())
				self._vw_AGENT = VideoWriter(self._exp_name, Utility.getVideoDir())
				self._vw_OCC = VideoWriter(self._exp_name, Utility.getVideoDir())

				if const.OCCUPANCY_MAP_MODE == const.VISITATION_MODE:
					self._vw_OCC = VideoWriter(self._exp_name, Utility.getVideoDir())
				elif const.OCCUPANCY_MAP_MODE == const.MOTION_MODE:
					self._vw_OCC_0 = VideoWriter(self._exp_name, Utility.getVideoDir())
					self._vw_OCC_1 = VideoWriter(self._exp_name, Utility.getVideoDir())
				else:
					Utility.die("Occupancy map mode not recognised in resetVideoWriters()", __file__)
		else:
			# Training data list to save upon completion (if we're even supposed to be
			# saving output at all)
			self._training_output = []

		# Class in charge of handling agent/targets
		self._object_handler = Object.ObjectHandler(	second_solver=second_solver, 
														dist_method=dist_method 		)

		# Class in charge of visitation map
		self._map_handler = VisitationMap.MapHandler(mark_visitation=mark_visitation)

		# Class in charge of visualisation (for both model input and our viewing benefit)
		self._visualiser = Visualisation.Visualiser(self._use_simulator)

	# Reset the map (agent position, target positions, memory, etc.)
	# Can supply function with epsiode configuration to override it
	def reset(self, a_pos=None, t_pos=None, m_pos=None):
		# Reset objects (agent, target), returns generated agent/target positions
		a_pos, t_pos, m_pos = self._object_handler.reset(a_pos=a_pos, t_pos=t_pos, m_pos=m_pos)
			
		# Extract the starting agent position
		a_x = a_pos[0]
		a_y = a_pos[1]

		# If we're using the gazebo simulator, move the agent/targets to generated positions
		if self._use_simulator:
			self._visualiser.resetAgentTargets(a_x, a_y, t_pos)

		# Reset the visitation map
		self._map_handler.reset(a_x, a_y)

		# If we should generate and save video
		if self._save_video: 
			self.resetVideoWriters()

		# Reset the algorithm
		if not self._generating: self._algorithm.reset()

		# Return generated epsidoe configuration
		return a_pos, t_pos, m_pos

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
		elif const.USE_EXT_ACTIONS and action == 'N': pass
		else: Utility.die("Action: {} not recognised!".format(action), __file__)

		# Is the agent now at a target position?
		target_match = False
		target_id = -1

		# Requested position is in bounds
		if Utility.checkPositionInBounds(req_x, req_y):
			# Set the new agent position
			unvisited, target_match, target_id = self._object_handler.setAgentPos(req_x, req_y)
		# Agent tried to move out of bounds, select a random valid action instead
		else:
			# Find possible actions from all actions given the map boundaries
			possible_actions = Utility.possibleActionsForPosition(old_x, old_y)

			# Randomly select an action
			rand_idx = random.randint(0, len(possible_actions)-1)
			choice = possible_actions[rand_idx]

			# Recurse to perform selected action
			return self.performAction(choice)

		# Update the map, function returns whether this new position
		# has been visited before
		is_new_location = self._map_handler.iterate(req_x, req_y, target_match, target_id)

		# self._map_handler.printMap()

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

	"""
	MAIN EPISODE LOOP BELOW
	"""

	# Begin this episode whether we're generating training data, testing, etc.
	def beginEpisode(self, testing, wait_amount=0, render_occ_map=False):
		# Render the initial episode state
		img, subview, occ_map0, occ_map1 = self._visualiser.update(self.retrieveStates(), render_occ_map=render_occ_map)

		# If we should generate and save video
		if self._save_video:
			self.iterateVideoWriters(img, subview, occ_map0, occ_map1)

		# Number of moves the agent has made
		num_moves = 0

		# Indicate to the solver to solve this episode/instance, returns the length
		# of the generated solution using the selected solver method
		# This is typically used for extracting the global optimum solution to a 
		# particular episode configuration
		sol_length, _ = self._object_handler.solveEpisode()

		# Display if we're supposed to
		if self._visualise: self._visualiser.display(wait_amount)

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

			# Iterate the object handler
			self._object_handler.iterate(num_moves)

			# Make the move
			_ = self.performAction(chosen_action)

			# Render the updated views (for input into the subsequent iteration)
			img, subview, occ_map0, occ_map1 = self._visualiser.update(self.retrieveStates(), render_occ_map=render_occ_map)

			# If we should generate and save video
			if self._save_video:
				self.iterateVideoWriters(img, subview, occ_map0, occ_map1, action=chosen_action)

			# Display if we're supposed to
			if self._visualise: self._visualiser.display(wait_amount)

			# Increment the move counter
			num_moves += 1

			# Ensure time hasn't expired yet
			if num_moves > const.MAX_NUM_MOVES:
				print "FAILED"
				break

		# If we're testing our algorithm
		if not self._generating:
			# Retrieve the number of loops detected. 0 for algorithms that don't use it
			num_loops = self._algorithm.getNumLoops()
		else: num_loops = 0

		# If we should generate and save video
		if self._save_video: 
			self.finishVideoWriters()

		# Finish up the object handler
		mu_DT = self._object_handler.finishUp()

		# Return the number of moves taken by the model and the solution
		# Also return the number of times loop detection is found
		# Finally return the average and stddev discovery per timestep rate
		return num_moves, sol_length, num_loops, mu_DT

	# For some timestep, append data to the big list
	def recordData(self, subview, visit_map, action_vector):
		# Create list of objects for this timestep
		row = [subview, visit_map, action_vector]

		# Add it to the list
		self._training_output.append(row)

	# Save output data to file
	def saveDataToFile(self):
		print "Saving data using h5py"

		file_str = "{}/TRAINING_DATA_{}.h5".format(Utility.getDataDir(), self._exp_name)

		# Open the dataset file (may overwrite an existing file!!)
		dataset = h5py.File(file_str, 'w')

		# The number of training instances generated
		num_instances = len(self._training_output)

		# The number of possible action classes
		if const.USE_EXT_ACTIONS:
			num_classes = len(const.EXT_ACTIONS)
		else:
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
		if const.OCCUPANCY_MAP_MODE == const.VISITATION_MODE:
			dataset.create_dataset('X1', (num_instances, const.MAP_WIDTH, const.MAP_HEIGHT))
		elif const.OCCUPANCY_MAP_MODE == const.MOTION_MODE:
			dataset.create_dataset('X1', (num_instances, const.MAP_WIDTH, const.MAP_HEIGHT, 2))
		dataset.create_dataset('Y', (num_instances, num_classes))

		# Actually add instances to the respective datasets
		for i in range(len(self._training_output)):
			dataset['X0'][i] = self._training_output[i][0]
			dataset['X1'][i] = self._training_output[i][1]
			dataset['Y'][i] = self._training_output[i][2]

		# Finish up
		dataset.close()

		print "Finished saving data at: {}".format(file_str)

		return file_str

	"""
	VideoWriter helper functions
	"""

	def resetVideoWriters(self):
		# Reset the other video writers if they exist
		if not self._generating:
			self._vw_OURS.reset("OURS")
			self._vw_AGENT.reset("AGENT_VISUAL")
			if const.OCCUPANCY_MAP_MODE == const.VISITATION_MODE:
				self._vw_OCC.reset("AGENT_OCC")
			elif const.OCCUPANCY_MAP_MODE == const.MOTION_MODE:
				self._vw_OCC_0.reset("AGENT_OCC_T")
				self._vw_OCC_1.reset("AGENT_OCC_A")
			else:
				Utility.die("Occupancy map mode not recognised in resetVideoWriters()", __file__)
		else:
			self._vw_GO_I.reset("GO_I")
			self._vw_GO_M.reset("GO_M")

	def iterateVideoWriters(self, img, subview, occ_map0, occ_map1, action=None):
		# If we're testing
		if not self._generating:
			self._vw_OURS.iterate(img, action=action)
			self._vw_AGENT.iterate(subview, action=action)
			if const.OCCUPANCY_MAP_MODE == const.VISITATION_MODE:
				self._vw_OCC.iterate(occ_map0, action=action)
			elif const.OCCUPANCY_MAP_MODE == const.MOTION_MODE:
				self._vw_OCC_0.iterate(occ_map0, action=action)
				self._vw_OCC_1.iterate(occ_map1, action=action)
			else:
				Utility.die("Occupancy map mode not recognised in iterateVideoWriters()", __file__)
		# We're generating training examples
		else:
			self._vw_GO_I.iterate(subview, action=action)
			self._vw_GO_M.iterate(occ_map0, action=action)

	def finishVideoWriters(self):
		# If we're testing
		if not self._generating:
			self._vw_OURS.finishUp()
			self._vw_AGENT.finishUp()
			if const.OCCUPANCY_MAP_MODE == const.VISITATION_MODE:
				self._vw_OCC.finishUp()
			elif const.OCCUPANCY_MAP_MODE == const.MOTION_MODE:
				self._vw_OCC_0.finishUp()
				self._vw_OCC_1.finishUp()
			else:
				Utility.die("Occupancy map mode not recognised in finishVideoWriters()", __file__)
		# We're generating training examples
		else:
			self._vw_GO_I.finishUp()
			self._vw_GO_M.finishUp()

	"""
	Experiment running functions
	"""

	def trainModel(self, experiment_name, data_dir):
		self._algorithm.trainModel(experiment_name, data_dir)

	# Do a given number of episodes
	def generateTrainingData(self, num_episodes):
		# Initialise progress bar (TQDM) object
		pbar = tqdm(total=num_episodes)

		for i in range(num_episodes):
			self.reset()
			self.beginEpisode(False, wait_amount=const.WAIT_AMOUNT)

			pbar.update()

		pbar.close()

		# Save the output if we're supposed to
		if self._save_output: 
			return self.saveDataToFile()

	# Do a given number of testing episodes
	def startTestingEpisodes(self, num_episodes):
		print "Beginning testing on generated examples"

		# Place to store testing data to in numpy form
		base = Utility.getDataDir()

		# Numpy array for testing data, consists of:
		# 0: number of moves required by the model
		# 1: number of moves required by employed solver (e.g. closest, target ordering)
		# 2: number of times loop detection is triggered
		# 3: Average discovery/per timestep for that episode
		test_data = np.zeros((num_episodes, 4))

		over_100 = 0
		opt = 0
		dif_10 = 0
		over_300 = 0
		average_DT = []

		# Initialise progress bar (TQDM) object
		pbar = tqdm(total=num_episodes)

		# Do some testing episodes
		for i in range(num_episodes):
			# Reset (generate a new episode)
			self.reset()

			# Go ahead and solve this instance using model & solver for comparison
			num_moves, sol_length, num_loops, mu_DT = self.beginEpisode(True, wait_amount=const.WAIT_AMOUNT)

			# Store statistics to numpy array
			test_data[i,0] = num_moves
			test_data[i,1] = sol_length
			test_data[i,2] = num_loops
			test_data[i,3] = mu_DT

			# Update stats
			if num_moves > 100: over_100 +=1
			if num_moves == sol_length: opt += 1
			if num_moves - sol_length <= 10: dif_10 += 1
			if num_moves > 300: over_300 += 1
			average_DT.append(mu_DT)

			# Compute percentages
			s1 = (float(over_100)/(i+1))*100
			s2 = (float(opt)/(i+1))*100
			s3 = (float(dif_10)/(i+1))*100
			s4 = (float(over_300)/(i+1))*100
			s5 = np.mean(np.asarray(average_DT))
			s6 = np.std(np.asarray(average_DT))

			# Print stats along the way
			print ">100={}%, opt={}%, <11 diff={}%, >300={}%, mu_DT={}, sigma_DT={}".format(s1, s2, s3, s4, s5, s6)

			# Update progress bar
			pbar.update()

		# Close up
		pbar.close()

		# Where to save numpy file to
		save_path = "{}/RESULTS_{}.npy".format(base, self._exp_name)

		# Save data to file
		np.save(save_path, test_data)

		# Print results
		ResultsHelper.listResults(save_path)

	# Compare solver performance over a number of testing episodes
	def compareSolvers(self, num_episodes):
		print "Beginning comparing solvers"

		# Place to store testing data to in numpy form
		base = Utility.getDataDir()

		# Numpy array for testing data, consists of:
		# 0: number of moves required by the model
		# 1: number of moves required by employed solver (e.g. closest, target ordering)
		# 2: number of times loop detection is triggered
		# 3: Average discovery/per timestep for that episode
		test_data = np.zeros((num_episodes, 4))

		# Initialise progress bar (TQDM) object
		pbar = tqdm(total=num_episodes) 

		# Do some comparisons
		for i in range(num_episodes):
			# Reset (generate a new episode)
			self.reset()

			# Get solution lengths (NS: naive solver, GO: globally-optimal)
			NS_length, mu_DT = self._object_handler.secondSolveEpisode()
			GO_length, _ = self._object_handler.solveEpisode()

			# Store statistics to numpy array
			test_data[i,0] = NS_length
			test_data[i,1] = GO_length
			test_data[i,2] = 0	# There's no loop detection here
			test_data[i,3] = mu_DT

			# Update progress bar
			pbar.update()

		# Close up
		pbar.close()

		# Where to save numpy file to
		save_path = "{}/RESULTS_{}.npy".format(base, self._exp_name)

		# Save data to file
		np.save(save_path, test_data)

		# Print results
		ResultsHelper.listResults(save_path)

	# Do a given number of episodes, saving video out to file
	def generateVideos(self, num_episodes, pause_beforehand=False):
		# Initialise progress bar
		pbar = tqdm(total=num_episodes)

		# If we're running the gazebo simulator, pause for user input
		if pause_beforehand:
			raw_input()

		# Iterate number of episodes times
		for i in range(num_episodes):
			"""
			Globally optimal solution
			"""
			self._generating = True
			# Generate a new episode
			a_pos, t_pos, m_pos = self.reset()

			# Record the solver's solution to the episode (globally-optimal)
			self.beginEpisode(False, wait_amount=const.WAIT_AMOUNT)

			"""
			Our solution
			"""
			self._generating = False
			# Supply the same starting configurations to the episode
			self.reset(a_pos=a_pos, t_pos=t_pos, m_pos=m_pos)
			# Record our solution to the episode
			self.beginEpisode(True, wait_amount=const.WAIT_AMOUNT, render_occ_map=True)

			# Update the progress bar
			pbar.update()

		# Close the progress bar
		pbar.close()

	# Generate solutions to randomly-generated episodes using trained model and output
	# visualisation of agent path to file if the difference between the generated
	# and globally-optimal solution are within a defined range
	def generateVisualisations(self, dif_range, num_images=25):
		print "Beginning generating visualisations to episodes"
		print "Trying to generate {} images in range {}".format(num_images, dif_range)

		# Load the DNN model from file
		self._dnn.loadSaveModel()

		# Place to store images to
		base = os.path.join(Utility.getFigureDir(), "raw_instances")

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

# Entry method/unit testing
if __name__ == '__main__':
	# Generate visualisations of runs for paper
	fm = FieldMap(visualise=True, use_simulator=False)
	fm.generateVisualisations((40, 50), num_images=25)
