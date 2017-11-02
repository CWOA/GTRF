#!/usr/bin/env python

import DNN
import Object
import Utility
import Visualisation
import VisitationMap

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
		self._loop_detector = Utility.LoopDetector()

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
		# Reset objects (agent, target)
		self._object_handler.reset()

		# Reset the visit map
		self._map_handler.reset()

		# Reset loop detection
		self._loop_detector.reset()

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

	def checkAgentInBounds(self):
		if self._agent_x < 0 or self._agent_y < 0:
			return False
		if self._agent_x >= self._grid_width or self._agent_y >= self._grid_height:
			return False

		return True

	def performAction(self, action):
		old_x, old_y = self._object_handler.getAgentPos()

		# Make the move
		if action == 'F': 	self._agent_y -= self._move_dist
		elif action == 'B': self._agent_y += self._move_dist
		elif action == 'L': self._agent_x -= self._move_dist
		elif action == 'R': self._agent_x += self._move_dist
		else: print "Action: {} not recognised!".format(action)

		if not self.checkAgentInBounds():
			self._agent_x = old_x
			self._agent_y = old_y

			# Find possible actions from all actions given the map boundaries
			possible_actions = self.checkMapBoundaries(list(self._actions))

			# Randomly select an action
			rand_idx = random.randint(0, len(possible_actions)-1)
			choice = possible_actions[rand_idx]

			# print "Agent trying to move out of bounds, randomly chose {}".format(choice)

			# Recurse
			return self.performAction(choice)

		# Check whether the agent has now visited a target
		ret = self.checkAgentTargetMatch()

		# Bool to store whether the agent has been to this location already
		has_visited = True

		# Check whether we've been here before
		if not self._map[self._agent_y, self._agent_x]:
			# Record history of map visitation
			self._map[self._agent_y, self._agent_x] = 1

			has_visited = False

		return has_visited, ret

	def beginInstance(self, testing, wait_amount=0):
		# Render the initial game state
		complete_img, subview = self._visualiser.update()

		# Number of moves the agent has made
		num_moves = 0

		# Display if we're supposed to
		if self._visualise: self._visualiser.display(wait_amount)

		# Indicator of whether the agent is stuck in an infinite loop
		if testing: agent_stuck = False

		# Loop until we've visited all the target objects
		while not self._object_handler.allTargetsVisited()
			# Use the DNN model to make action decisions
			if testing:
				# Get the map
				visit_map = self._map_handler.getMap()

				# Use DNN model to predict correct action
				action_vector = self._dnn.testModelSingle(subview, visit_map)

				# Convert to action
				chosen_action = Utility.classVectorToAction(action_vector)

				# Add the suggested action and check history, check if the agent is
				# stuck in a loop, act accordingly
				if not agent_stuck and self._loop_detector.addCheckAction(chosen_action):
					agent_stuck = True

				# Agent is stuck, move towards nearest unvisited location
				if agent_stuck:
					a_x, a_y = self._object_handler.getAgentPos()
					action = self._map_handler.findUnvisitedDirection(a_x, a_y)

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
			new_location = self.performAction(chosen_action)

			# Check whether the agent is still stuck
			if agent_stuck and testing and new_location: agent_stuck = False

			# Increment the move counter
			num_moves += 1

			# Render the updated view
			complete_img, subview = self._visualiser.update()

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
			self.reset()
			self.beginInstance(False)

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
			self.reset()
			num_moves = self.beginInstance(True)

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
	
