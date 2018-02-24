#!/usr/bin/env python

# Core libraries
import sys
sys.path.append('../')
import math
import time
import copy
import random
import numpy as np
from tqdm import tqdm

# My libraries/classes
import Constants as const
from Solvers.Solver import EpisodeSolver
from Utilities.DiscoveryRate import DiscoveryRate
from Utilities.Utility import Utility

"""
TBC
"""

class ObjectHandler:
	# Class constructor
	def __init__(	self,
					random_agent_pos=const.RANDOM_AGENT_START_POS,
					random_num_targets=False,
					solver_method=const.SOLVER_METHOD,
					dist_method=const.OBJECT_DIST_METHOD,
					second_solver=False,
					individual_motion=const.INDIVIDUAL_MOTION,
					motion_method=const.INDIVIDUAL_MOTION_METHOD	):
		"""
		Class arguments from init
		"""

		# Should the agent's position be randomised or some default position
		self._random_agent_pos = random_agent_pos

		# Should we randomise the number of targets or not
		self._random_num_targets = random_num_targets

		# How should object's be initialised spatially?
		self._object_dist_method = dist_method

		# Should each individual move per iteration according to their own velocity
		# and heading parameters
		self._individual_motion = individual_motion

		# If motion is enabled, how should individuals move
		self._motion_method = motion_method

		"""
		Class attributes
		"""

		# Class responsible for selecting agent actions towards solving a given episode
		self._solver = EpisodeSolver(solver_method)

		# Are we initialising another solver
		self._use_second_solver = second_solver

		# Second solver class (for comparisons between solution methods)
		if self._use_second_solver:
			self._second_solver = EpisodeSolver(const.NAIVE_SOLVER)

		# For handling target discovery rate statistics
		self._dr = DiscoveryRate()

		# Agent and list of targets
		self._agent = None
		self._targets = None

	# Reset this handler so we can go again
	def reset(self, a_pos=None, t_pos=None):
		# Unique identifier (ID) counter for objects (both agent and targets)
		self._id_ctr = 0

		# Generate a random starting agent coordinate if we're supposed to
		if self._random_agent_pos:
			# If we've been given agent starting coordinates
			if a_pos is not None:
				a_x = a_pos[0]
				a_y = a_pos[1]
			# Generate a new random position in the environment
			else:
				a_x = random.randint(0, const.MAP_WIDTH-1)
				a_y = random.randint(0, const.MAP_HEIGHT-1)

			# Create the agent object with these starting coordinates
			self._agent = Object(self._id_ctr, True, x=a_x, y=a_y)

		# Default agent starting coordinates
		else: self._agent = Object(self._id_ctr, True)

		# Increment the ID counter
		self._id_ctr += 1

		# Number of target objects to generate
		num_targets = const.NUM_TARGETS

		# Randomise the number of targets to generate if we're supposed to
		if self._random_num_targets:
			num_targets = rand.randint(const.NUM_TARGETS_RANGE[0], const.NUM_TARGETS_RANGE[1])

		# Initialise the targets list
		self._targets = []

		# Precompute object positions for some distribution types (e.g. grids)
		self.preComputeObjectPositions()

		# Generate the targets
		for i in range(num_targets):
			# If we've been given target starting positions
			if t_pos is not None:
				t_x = t_pos[i][0]
				t_y = t_pos[i][1]
			# Generate new target position
			else:
				t_x, t_y = self.generateUnoccupiedPosition()

			# Create the target object instance
			target = Object(	self._id_ctr, 
								False, 
								x=t_x, 
								y=t_y,
								individual_motion=self._individual_motion,
								motion_method=self._motion_method			)

			self._targets.append(target)
			self._id_ctr += 1

		# Reset equidistant parameters
		self._equi_x = []
		self._equi_y = []

		# In case motion is disabled
		self._motion_pos = None

		# If random walk individual motion is enabled
		if self._individual_motion:
			# Pre-determine random walks so that we can generate GO solution
			if self._motion_method == const.INDIVIDUAL_MOTION_RANDOM:
				self._motion_pos, success = self.predetermineMotionWalk(False, const.MOTION_NUM_STEPS)
			elif self._motion_method == const.INDIVIDUAL_MOTION_HERD:
				self._motion_pos, success = self.predetermineMotionWalk(True, const.MOTION_NUM_STEPS)

			# If we coudln't figure out non-collision random walk for this configuration
			# try generating another (recurse)
			if not success:
				print "Resetting"
				return self.reset()
			# Otherwise assign random walk positions to each target
			else:
				for i in range(len(self._targets)):
					self._targets[i].assignRandomWalk(self._motion_pos, i)

		# Give generated agent and target objects to the solver
		self._solver.reset(		copy.deepcopy(self._agent), 
								copy.deepcopy(self._targets), 
								rand_pos=self._motion_pos 		)

		# Initialise the second solver if we're supposed to
		if self._use_second_solver:
			self._second_solver.reset(	copy.deepcopy(self._agent), 
										copy.deepcopy(self._targets)	)

		# Reset the DT metric
		self._dr.reset()

		return self.getAgentPos(), self.getTargetPositions()

	# Called at each iteration, just used for individual and population motion (if enabled)
	def iterate(self, itr):
		# If individual-motion is enabled, enact it
		if self._individual_motion:
			for t in self._targets:
				t.step(itr)

		# Update the target discovery per timestep metric
		self._dr.iterate()

	# Called at the end of each episode
	def finishUp(self):
		# Finish up the DT metric for this episode
		mu_DT = self._dr.finish()

		return mu_DT

	"""
	Object-centric methods
	"""

	# For some target distribution types, need to generate all values beforehand
	# (not per individual)
	def preComputeObjectPositions(self):
		# If targets should be equidistant
		if (self._object_dist_method == const.STAT_DIST or
			self._object_dist_method == const.EQUI_DIST		):
			# x, y array with boundaries and spacing
			x = np.arange(const.STAT_START_X, const.MAP_WIDTH, const.GRID_SPACING)
			y = np.arange(const.STAT_START_Y, const.MAP_HEIGHT, const.GRID_SPACING)

			# Create two coordinate arrays with this
			equi_x, equi_y = np.meshgrid(x, y)

			# Offset odd rows of x by half the spacing size to yield approximate 
			# equidistant spacing
			half_spacing = round(const.GRID_SPACING/2)
			for i in range(equi_x.shape[0]):
				for j in range(equi_x.shape[1]):
					# If i is odd
					if i % 2 == 1:
						equi_x[i,j] += half_spacing

			# Convert to python list
			self._equi_x = equi_x.flatten().tolist()
			self._equi_y = equi_y.flatten().tolist()

		# If the grid is supposed to move around
		if self._object_dist_method == const.EQUI_DIST:
			# Manually fill grid positions ()
			self._equi_x = [0, 2, 4, 1, 3, 5]
			self._equi_y = [0, 0, 0, 2, 2, 2]

			# Detemine random grid offsets
			off_x = random.randint(0, const.MAP_WIDTH-6)
			off_y = random.randint(0, const.MAP_HEIGHT-3)

			# Apply the offsets to each grid coordinate
			assert len(self._equi_x) == len(self._equi_y)
			for i in range(len(self._equi_x)):
				# Apply the offset
				self._equi_x[i] += off_x
				self._equi_y[i] += off_y

	# Generate a random position within the grid that isn't already occupied
	def generateUnoccupiedPosition(self):
		# List of occupied positions
		occupied = []

		# Combine all generated positions up to this point
		occupied.append(self.getAgentPos())
		if self._targets is not None:
			for target in self._targets:
				occupied.append(target.getPosTuple())

		# Loop until we've generated a valid position
		while True:
			# Python "random" class uses PRNG Mersenne Twister
			if self._object_dist_method == const.PRNG_DIST:
				# Generate a position within bounds
				x = random.randint(0, const.MAP_WIDTH-1)
				y = random.randint(0, const.MAP_HEIGHT-1)
			# Equidistant targets
			elif (self._object_dist_method == const.STAT_DIST or
				  self._object_dist_method == const.EQUI_DIST	):
				# Pop the next x,y coordinates from the list
				x = self._equi_x.pop(0)
				y = self._equi_y.pop(0)
			# Use a Gaussian distribution
			elif self._object_dist_method == const.GAUS_DIST:
				x = int(round(random.gauss(const.GAUS_MU_X, const.GAUS_SIGMA_X)))
				y = int(round(random.gauss(const.GAUS_MU_Y, const.GAUS_SIGMA_Y)))
			else:
				Utility.die("Object distribution method not recognised in generateUnoccupiedPosition()", __file__)

			# Generated position is valid
			ok = True

			# Check the generated position isn't already in use
			for pos in occupied:
				if x == pos[0] and y == pos[1]:
					ok = False
					break

			# Check the generated position is within the map bounds
			if x < 0 or y < 0 or x >= const.MAP_WIDTH or y >= const.MAP_HEIGHT:
				ok = False

			if ok: return x, y

	# Print coordinates of the agent and all targets
	def printObjectCoordinates(self):
		a_x, a_y = self._agent.getPos()
		print "Agent pos = ({},{})".format(a_x, a_y)

		for i in range(len(self._targets)):
			t_x, t_y = self._targets[i].getPos()
			print "Target #{} pos = ({},{})".format(i, t_x, t_y)

	# Update the position of the agent
	def updateAgentPos(self, x, y):
		self._agent.setPos(x, y)

	# Simply returns the position of the agent
	def getAgentPos(self):
		return self._agent.getPos()

	# Set the position of agent, check whether it matches a target position
	def setAgentPos(self, x, y):
		# Set the position
		self._agent.setPos(x, y)

		return self.checkAgentTargetMatch(x, y)

	# Marks a target as visited if the agent position matches it
	def checkAgentTargetMatch(self, a_x, a_y):
		for target in self._targets:
			t_x, t_y = target.getPos()

			if a_x == t_x and a_y == t_y:
				if not target.getVisited():
					# Update the discovery per timestep value
					self._dr.discovery()

					# Indicate that we've not visited this target
					target.setVisited(True)

					# Return that this is the case and the target's ID
					return True, target.getID()

		return False, -1

	# Returns a list of all target positions
	def getTargetPositions(self):
		positions = []

		for target in self._targets:
			positions.append(target.getPosTuple())

		return positions

	# Returns target positions as a list of lists (instead of tuple)
	def getTargetPositionsList(self):
		positions = []

		for target in self._targets:
			positions.append(target.getPosList())

		return positions

	# Returns True if all targets have been visited
	def allTargetsVisited(self):
		for target in self._targets:
			if not target.getVisited():
				return False

		return True

	# If individual motion and random/herd walk is enabled, this function pre-determines the
	# random motion of the targets in a way that object-object collisions are avoided
	# such that a globally-optimal solution can be generated
	def predetermineMotionWalk(self, herd, num_steps):
		# Get the current target positions
		cur_pos = self.getTargetPositions()

		# Number of attempts to make to randomly generate random positions with no
		# collisions
		attempts_threshold = 10

		if herd:
			# Choose a random overall herd direction
			direction = random.choice(const.ACTIONS)

		# List of random walk positions
		rand_pos = []
		#rand_pos.append(cur_pos)

		# Generate num_steps random walks
		for i in range(num_steps):
			# Make a copy of the positions
			new_pos = list(cur_pos)

			# Only move every velocity steps
			if i % const.INDIVIDUAL_VELOCITY == 0:
				# Number of attempts at generating non-collision positions
				attempts = 0

				# Loop until there are no collisions
				while True:
					# Loop over the number of targets
					for j in range(const.NUM_TARGETS):
						# The list of all actions possible in this position
						possible_actions = Utility.possibleActionsForPosition(new_pos[j][0], new_pos[j][1])

						# If herd-like dynamics
						if herd:
							# Generate a float in range [0,1]
							chance = random.uniform(0, 1)

							# Chance of applying random action that isn't the current direction
							# or the opposite direction
							if chance <= const.INDIVIDUAL_RANDOM_CHANCE:
								direction_removed = list(const.EXT_ACTIONS)
								direction_removed.remove(direction)
								direction_removed.remove(Utility.oppositeAction(direction))
								action = random.choice(direction_removed)
							# If the group heading is possible, apply it
							elif direction in possible_actions:
								action = direction
							# Direction isn't possible (we're probably at the map boundaries)
							else:
								# pick a new random direction that isn't what we've just done
								direction_removed = list(const.ACTIONS)
								direction_removed.remove(direction)
								direction = random.choice(direction_removed)

								action = 'N'

							# Apply the action selection
							if action == 'F': t_pos = (new_pos[j][0], new_pos[j][1] - 1)
							elif action == 'B': t_pos = (new_pos[j][0], new_pos[j][1] + 1)
							elif action == 'L': t_pos = (new_pos[j][0] - 1, new_pos[j][1])
							elif action == 'R': t_pos = (new_pos[j][0] + 1, new_pos[j][1])
							elif action == 'N': t_pos = new_pos[j]

							# Assign the tuple back
							new_pos[j] = t_pos

						# Implement random walk
						else:
							# Shuffle the list
							random.shuffle(possible_actions)

							# Loop over possible shuffled actions for this position
							for action in possible_actions:
								# Apply the action
								if action == 'F': t_pos = (new_pos[j][0], new_pos[j][1] - 1)
								elif action == 'B': t_pos = (new_pos[j][0], new_pos[j][1] + 1)
								elif action == 'L': t_pos = (new_pos[j][0] - 1, new_pos[j][1])
								elif action == 'R': t_pos = (new_pos[j][0] + 1, new_pos[j][1])

								# Assign the tuple back
								new_pos[j] = t_pos

								# print "Inside action={}, attempts={}".format(action, attempts)

								# Check the position is in the map boundaries
								if Utility.checkPositionInBounds(new_pos[j][0], new_pos[j][1]):
									break

					# Ensure there are no duplicate generated positions
					assert(len(new_pos) == const.NUM_TARGETS)
					if len(new_pos) == len(set(new_pos)):
						# Check all positions are in map boundaries
						if Utility.checkPositionsListInBounds(new_pos):
							break

					# We've tried again to generate non-collision positions
					attempts += 1

					# If we've exceeded a threshold, there's probably no solution
					if attempts >= attempts_threshold: return None, False

			# If we're here, generated positions are valid, add to the list
			rand_pos.append(new_pos)

			# Update current pos
			cur_pos = new_pos

		return rand_pos, True

	"""
	Episode Solver methods
	"""

	# Simply signal the solver to solve this episode, returns the length of the solution
	def solveEpisode(self):
		return self._solver.solveEpisode()

	# Solve episode using second solver method
	def secondSolveEpisode(self):
		return self._second_solver.solveEpisode()

	# Simply get the next action from the solver
	def nextSolverAction(self):
		return self._solver.getNextAction()

	# Method for comparing the performance of solver methods
	# 
	# This particular one is for comparing time required to generate solutions
	# as well as the number of moves
	def determineSolverStats(self, load=False):
		# Initialise instances of different solver methods
		seq_solver = EpisodeSolver(const.SEQUENCE_SOLVER)
		clo_solver = EpisodeSolver(const.CLOSEST_SOLVER)

		# Number of instances to run per change of the number of targets
		inst = 100

		# Number of targets to start and end with
		start_num_targets = 2
		end_num_targets = 9

		num_targets = range(start_num_targets, end_num_targets)

		# Difference
		dif_num_targets = len(num_targets)

		# Get base directory(folder) to save numpy arrays to
		base = Utility.getICIPDataDir()

		# Need to generate data
		if not load:
			# Intialise progress bar (TQDM)
			pbar = tqdm(total=inst*dif_num_targets)

			# Numpy data arrays for time required and number of moves
			time_seq = np.zeros((dif_num_targets, inst))
			time_clo = np.zeros((dif_num_targets, inst))
			moves_seq = np.zeros((dif_num_targets, inst))
			moves_clo = np.zeros((dif_num_targets, inst))

			for i in range(len(num_targets)):
				for j in range(inst):
					# Change the number of targets
					const.NUM_TARGETS = num_targets[i]

					# Generate a random new instance and give to solvers
					self.reset()
					seq_solver.reset(copy.deepcopy(self._agent), copy.deepcopy(self._targets))
					clo_solver.reset(copy.deepcopy(self._agent), copy.deepcopy(self._targets))

					# Time the sequence solver
					tic = time.clock()
					m0 = seq_solver.solveEpisode()
					t0 = time.clock() - tic

					# Time the closest solver
					tic = time.clock()
					m1 = clo_solver.solveEpisode()
					t1 = time.clock() - tic

					# Add values to data matrices
					time_seq[i, j] = t0
					time_clo[i, j] = t1
					moves_seq[i, j] = m0
					moves_clo[i, j] = m1

					# Update the progress bar
					pbar.update()

			pbar.close()

			# Save numpy data arrays to file
			np.save("{}/time_seq".format(base), time_seq)
			np.save("{}/time_clo".format(base), time_clo)
			np.save("{}/moves_seq".format(base), moves_seq)
			np.save("{}/moves_clo".format(base), moves_clo)

		# Just load data
		else:
			time_seq = np.load("{}/time_seq.npy".format(base))
			time_clo = np.load("{}/time_clo.npy".format(base))
			moves_seq = np.load("{}/moves_seq.npy".format(base))
			moves_clo = np.load("{}/moves_clo.npy".format(base))

		# Use utility functions to draw the graph
		# Utility.drawGenerationTimeGraph(	time_seq, 
		# 									time_clo, 
		# 									start_num_targets, 
		# 									end_num_targets			)

		Utility.drawGenerationLengthGraph(	moves_seq,
											moves_clo,
											start_num_targets,
											end_num_targets			)
		# Utility.drawGenerationGraphs(	moves_seq, 
		# 								moves_clo,
		# 								time_seq,
		# 								time_clo,
		# 								num_targets 				)

class Object:
	# Class constructor
	def __init__(	self,
					ID,
					is_agent, 
					x=const.AGENT_START_COORDS[0], 
					y=const.AGENT_START_COORDS[1],
					individual_motion=const.INDIVIDUAL_MOTION,
					motion_method=const.INDIVIDUAL_MOTION_METHOD	):
		"""
		Class attributes/properties
		"""

		# Unique object ID (for both agent and targets)
		self._ID = ID

		# Am I the agent or a target?
		self._agent = is_agent

		# Object position (x, y) coordinates
		self._x = x 
		self._y = y

		# Variables depending on whether we're the agent
		if self._agent:
			# Colour to render for visualisation
			self._colour = const.AGENT_COLOUR
		# We're a target
		else:
			# As a target, have we been visited before
			self._visited = False

			# Colour to render for visualisation
			self._colour = const.TARGET_COLOUR

			# Are we supposed to move every so often
			self._motion = individual_motion

			if self._motion:
				# The selected motion model
				self._motion_method = motion_method

				# If this target is using the heading/velocity model
				if self._motion_method == const.INDIVIDUAL_MOTION_HEADING:
					# Randomly choose a heading
					rand_heading = random.randint(0, 360)
					self._heading = math.radians(rand_heading)

					# Target velocity in grid squares per iteration
					self._velocity = 0.5
				# Random walk or herd movement
				elif (	self._motion_method == const.INDIVIDUAL_MOTION_RANDOM or
						self._motion_method == const.INDIVIDUAL_MOTION_HERD			):
					# List of random walk coordinates per iteration
					self._object_walk = []

	# Class to string method
	def __str__(self):
		if self._agent:
			obj_type = "Agent"
		else:
			obj_type = "Target"

		return "{}: ({},{})".format(obj_type, self._x, self._y)

	# Allow the object to move according to its motion parameters
	def step(self, itr):
		# Ensure we're a target
		assert(not self._agent)

		# Ensure we're supposed to move
		assert(self._motion)

		# Just replay random walk from pre-determined list
		if (	self._motion_method == const.INDIVIDUAL_MOTION_RANDOM or
				self._motion_method == const.INDIVIDUAL_MOTION_HERD			):
			pos = self._object_walk.pop(0)
			self._x = pos[0]
			self._y = pos[1]
		# If this target is using the heading/velocity model
		elif self._motion_method == const.INDIVIDUAL_MOTION_HEADING:
			# If we're supposed to move this iteration (due to our velocity constant)
			if itr % const.INDIVIDUAL_VELOCITY == 0:
				# Compute x,y movement vector
				d_x = self._velocity * math.sin(self._heading)
				d_y = self._velocity * math.cos(self._heading)

				# Enact this vector to the current target position
				self._x += d_x
				self._y += d_y
		else:
			Utility.die("Given motion model does not exist (look in Constants.py)", __file__)

	# Returns a DEEP copy of this object
	def copy(self):
		return copy.deepcopy(self)

	def performAction(self, action):
		if self._agent:
			if action == 'F': 	self._y -= const.MOVE_DIST
			elif action == 'B': self._y += const.MOVE_DIST
			elif action == 'L': self._x -= const.MOVE_DIST
			elif action == 'R': self._x += const.MOVE_DIST
		else:
			Utility.die("Trying to perform action on a non-agent", __file__)

	# Extract this target's random walk from the given list of lists of tuples
	def assignRandomWalk(self, rand_pos, i):
		# Loop over the number of steps
		for j in range(len(rand_pos)):
			self._object_walk.append(rand_pos[j][i])

	"""
	Getters
	"""
	def getID(self):
		return self._ID
	def getVisited(self):
		return self._visited
	def getPos(self):
		return self._x, self._y
	def getPosTuple(self):
		return (self._x, self._y)
	def getPosList(self):
		return [self._x, self._y]
	def getColour(self):
		return self._colour
	def isAgent(self):
		return self._agent
	def getPosTupleAtTimestep(self, timestep):
		if self.isAgent():
			return self.getPosTuple()
		else:
			return self._object_walk[timestep-1]

	"""
	Setters
	"""
	def setVisited(self, visited):
		if not self._agent:
			self._visited = visited
		else:
			Utility.die("Trying to set agent visitation", __file__)
	def setPos(self, x, y):
		if self._agent:
			self._x = x
			self._y = y
		else:
			Utility.die("Trying to directly set position of non-agent", __file__)

# Entry method for unit testing
if __name__ == '__main__':
	object_handler = ObjectHandler()
	object_handler.determineSolverStats(load=True)
