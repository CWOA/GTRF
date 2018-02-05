#!/usr/bin/env python

import time
import copy
import random
import numpy as np
from tqdm import tqdm
import Constants as const
from Utility import Utility
from Solvers.Solver import EpisodeSolver


class ObjectHandler:
	# Class constructor
	def __init__(	self,
					random_agent_pos=True,
					random_num_targets=False,
					solver_method=const.SOLVER_METHOD,
					dist_methood=const.OBJECT_DIST_METHOD,
					second_solver=False				 		):
		"""
		Class arguments from init
		"""

		# Should the agent's position be randomised or some default position
		self._random_agent_pos = random_agent_pos

		# Should we randomise the number of targets or not
		self._random_num_targets = random_num_targets

		# How should object's be initialised spatially?
		self._object_dist_method = dist_methood

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

		# Pointers to agent and list of targets
		self._agent = None
		self._targets = None

		print "Initialised ObjectHandler"

	# Reset this handler so we can go again
	def reset(self):
		# Unique identifier (ID) counter for objects (both agent and targets)
		self._id_ctr = 0

		# Generate a random starting agent coordinate if we're supposed to
		if self._random_agent_pos:
			# MIGHT NEED CHANGING: based on object distribution method
			# possible incorporate within "generateUnoccupiedPosition()" below? 
			a_x = random.randint(0, const.MAP_WIDTH-1)
			a_y = random.randint(0, const.MAP_HEIGHT-1)
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

		# Generate the targets
		for i in range(num_targets):
			t_x, t_y = self.generateUnoccupiedPosition()
			self._targets.append(Object(self._id_ctr, False, x=t_x, y=t_y))
			self._id_ctr += 1

		# Give generated agent and target objects to the solver
		self._solver.reset(copy.deepcopy(self._agent), copy.deepcopy(self._targets))

		# Initialise the second solver if we're supposed to
		if self._use_second_solver:
			self._second_solver.reset(copy.deepcopy(self._agent), copy.deepcopy(self._targets))

		return self.getAgentPos(), self.getTargetPositions()

	"""
	Object-centric methods
	"""

	# Generate a random position within the grid that isn't already occupied
	def generateUnoccupiedPosition(self):
		# List of occupied positions
		occupied = []

		# Combine all generated positions up to this point
		occupied.append(self.getAgentPos())
		if self._targets is not None:
			for target in self._targets:
				occupied.append(target.getPosTuple())

		# If targets should be equidistant
		if self._object_dist_method == const.EQUI_DIST:
			# x, y array with boundaries and spacing
			x = np.arange(const.EQUI_START_X, const.MAP_WIDTH, const.EQUI_SPACING)
			y = np.arange(const.EQUI_START_Y, const.MAP_HEIGHT, const.EQUI_SPACING)

			# Create two coordinate arrays with this
			equi_x, equi_y = np.meshgrid(x, y)

			# Offset odd rows of x by half the spacing size to yield equidistant spacing
			half_spacing = round(const.EQUI_SPACING/2)
			for i in range(equi_x.shape[0]):
				for j in range(equi_x.shape[1]):
					# If i is odd
					if i % 2 == 1:
						equi_x[i,j] += half_spacing

			# Convert to python list
			equi_x = equi_x.flatten().tolist()
			equi_y = equi_y.flatten().tolist()

		# Loop until we've generated a valid position
		while True:
			# Python "random" class uses PRNG Mersenne Twister
			if self._object_dist_method == const.PRNG_DIST:
				# Generate a position within bounds
				x = random.randint(0, const.MAP_WIDTH-1)
				y = random.randint(0, const.MAP_HEIGHT-1)
			# Equidistant targets
			elif self._object_dist_method == const.EQUI_DIST:
				# Pop the next x,y coordinates from the list
				x = equi_x.pop(0)
				y = equi_y.pop(0)
			# Use a Gaussian distribution
			elif self._object_dist_method == const.GAUS_DIST:
				# Gaussian parameters are constant, is this ok?
				x = int(round(random.gauss(const.GAUS_MU_X, const.GAUS_SIGMA_X)))
				y = int(round(random.gauss(const.GAUS_MU_Y, const.GAUS_SIGMA_Y)))
			else:
				Utility.die("Object distribution method not recognised", __file__)

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

		self.checkAgentTargetMatch(x, y)

	# Marks a target as visited if the agent position matches it
	def checkAgentTargetMatch(self, a_x, a_y):
		for target in self._targets:
			t_x, t_y = target.getPos()

			if a_x == t_x and a_y == t_y:
				if not target.getVisited():
					target.setVisited(True)

	# Returns a list of all target positions
	def getTargetPositions(self):
		positions = []

		for target in self._targets:
			positions.append(target.getPosTuple())

		return positions

	# Returns True if all targets have been visited
	def allTargetsVisited(self):
		for target in self._targets:
			if not target.getVisited():
				return False

		return True

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

		# Utility.drawGenerationLengthGraph(	moves_seq,
		# 									moves_clo,
		# 									start_num_targets,
		# 									end_num_targets			)
		Utility.drawGenerationGraphs(	moves_seq, 
										moves_clo,
										time_seq,
										time_clo,
										num_targets 				)

class Object:
	# Class constructor
	def __init__(	self,
					ID,
					is_agent, 
					x=const.AGENT_START_COORDS[0], 
					y=const.AGENT_START_COORDS[1]	):
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

		# Variables depending on whetherh we're the agent
		if self._agent:
			# Colour to render for visualisation
			self._colour = const.AGENT_COLOUR
		else:
			# As a target, have we been visited before
			self._visited = False

			# Colour to render for visualisation
			self._colour = const.TARGET_COLOUR

	# Class to string method
	def __str__(self):
		if self._agent:
			obj_type = "Agent"
		else:
			obj_type = "Target"

		return "{}: ({},{})".format(obj_type, self._x, self._y)

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
			Utility.die("Trying to perform action on a non-agent")		

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
	def getColour(self):
		return self._colour
	def isAgent(self):
		return self._agent

	"""
	Setters
	"""
	def setVisited(self, visited):
		if not self._agent:
			self._visited = visited
		else:
			Utility.die("Trying to set agent visitation")

	def setPos(self, x, y):
		if self._agent:
			self._x = x
			self._y = y
		else:
			Utility.die("Trying to directly set position of non-agent")

# Entry method for unit testing
if __name__ == '__main__':
	object_handler = ObjectHandler()
	object_handler.determineSolverStats(load=True)
