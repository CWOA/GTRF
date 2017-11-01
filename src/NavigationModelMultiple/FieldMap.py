#!/usr/bin/env python

import Object

class FieldMap:
	# Class constructor
	def __init__(		self, 
						visualise=False, 
						agent_global_view=True, 
						random_agent_pos=True, 
						save=False						):
		### Class attributes

		self._object_handler = Object.ObjectHandler()

		# Whether or not we should be saving output to file
		self._save_output = save

		# Training data list to pickle upon completion
		self._training_output = []

		# Which training data generation method to use
		# see begin() function
		self._agent_has_global_view = agent_global_view

		# Position of agent should be generated randomly
		self._random_agent_pos = random_agent_pos

		# Unit to move agent by each movement
		self._move_dist = 1

		# Number of target objects (number of cows)
		self._num_targets = 5

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

		# Dimensions of display/visualisation grid
		self._disp_width = self._grid_pixels * self._grid_width
		self._disp_height = self._grid_pixels * self._grid_height

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

	def checkAgentInBounds(self):
		if self._agent_x < 0 or self._agent_y < 0:
			return False
		if self._agent_x >= self._grid_width or self._agent_y >= self._grid_height:
			return False

		return True

	def performAction(self, action):
		old_x = self._agent_x
		old_y = self._agent_y

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

			# Random 0 element in subview
			size = indices[1].shape[0]
			rand_element = random.randint(0, size-1)
			z_x = indices[1][rand_element]
			z_y = indices[0][rand_element]

			# 0 element position in subview
			# z_x = indices[1][0]
			# z_y = indices[0][0]

			# Find the best action for the angle between them
			action = self.findActionForAngle((a_x, a_y), (z_x, z_y))

		# print sub
		# print indices
		# cv2.waitKey(0)

		return action

	def findUnvisitedDirection(self):
		# Cell search radius for unvisited cells
		radius = 1

		# Determined action to take
		action = None

		# Loop until we find a suitable unvisited direction
		while action is None:
			# Try and find an unvisited location in the current radius
			action = self.determineCellNeighbours(self._agent_x, self._agent_y, radius)

			# Increment the radius
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
			has_visited, reward = self.performAction(chosen_action)
			num_visited += reward

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