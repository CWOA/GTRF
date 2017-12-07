#!/usr/bin/env python

import sys
import copy
import time
import random
import numpy as np
import networkx as nx
import Constants as const
from Utility import Utility
import matplotlib.pyplot as plt

"""
Globally-optimal solution constructs a directed tree of all possible moves for a given
agent starting position. The issue is the combinatorial explosion that occurs when 
increasing the size of the grid to anything reasonable. The complexity of this solution
is therefore dependent on the size of the grid (amongst other parameters)
"""

# Class is a subclass of the "Solver" superclass
class TreeSolver:
	# Class constructor
	def __init__(	self,
					m_w,
					m_h,
					manual_position=False):
		# Initialise superclass


		"""
		Class setup
		"""

		# Setup class parameters/attributes
		self.reset(manual_position, m_w, m_h)

	def reset(self, manual_position, m_w, m_h):
		"""
		Class attributes
		"""

		# We want to manually specify starting conditions (mostly for bug fixing)
		if manual_position:
			self._map_w = 8
			self._map_h = 8
			self._init_x = 6
			self._init_y = 7
			self._targets = [(2, 6), (7, 4), (4, 2), (4, 1), (7, 0)]
			const.NUM_TARGETS = len(self._targets)
		else:
			# Environment dimensions
			self._map_w = m_w
			self._map_h = m_h

			# Initial agent coordinates
			self._init_x, self._init_y = self.generateMapPosition()

			# Randomly generate target positions
			self._targets = self.generateTargets(self._init_x, self._init_y)

		# Actual directed graph/tree for exploration
		self._graph = nx.DiGraph()

		# Unique identifier counter for each node
		self._id_ctr = 0

		# Stores the current best number of timesteps
		self._best_depth = sys.maxint

		# Print some stats
		print "Map dimensions = {}*{}".format(self._map_w, self._map_h)
		print "Agent position = ({},{})".format(self._init_x, self._init_y)
		print "Target positions = {}".format(self._targets)

	"""
	Tree construction methods
	"""

	# Create the root node and begin growing the tree recursively
	def beginGrowingTree(self):
		# Create root node attributes
		root_attr = NodeAttributes(		self._id_ctr, 
										self._init_x, 
										self._init_y,
										self._map_w,
										self._map_h,
										colour=const.ROOT_NODE_COLOUR	)

		# Create root node within the graph
		self._graph.add_node(self._id_ctr, attr=root_attr)

		# Increment the node ID counter
		self._id_ctr += 1

		# Generate the actual tree
		tic = time.clock()
		self.growTree(root_attr)
		toc = time.clock()

		# Total time required (seconds)
		total = toc - tic

		print "Time required = {} seconds".format(total)

		return total

	# Recursive method for growing the tree/graph
	def growTree(self, parent_attr):
		# Find possible actions for the parent node's current position
		actions = parent_attr.possibleActions()

		# Only perform this if there's more than one possible action in this state
		if len(actions) > 1:
			# Re-order actions by closest unvisited targets to encourage finding solutions
			# early and avoiding unnecessary recursion in the future
			actions = parent_attr.reorderActions(actions, self._targets)

		# Loop over possible actions
		for action in actions:
			# Create a child attribute with the action enacted
			curr_attr = parent_attr.newUpdatedInstance(action, self._id_ctr)

			# Create node in the graph
			self.addNode(parent_attr, curr_attr, action)

			# Increment the node counter
			self._id_ctr += 1

			# print self._id_ctr

			# print action
			# raw_input()

			# See whether this new position visits unvisited targets
			all_visited = curr_attr.checkTargets(self._targets)

			# See how many timesteps this child has made without visiting a new target
			tsv = curr_attr.getTimeSinceVisit()

			# If this child hasn't visited all targets and hasn't made too many moves
			# without visiting a new target
			if not all_visited:
				# Only recurse if this path has visited a new target in the near past
				if tsv < const.MAX_TIME_SINCE_VISIT:
					# Recurse if it hasn't exceeded the depth of the current best solution
					if curr_attr.getT() < self._best_depth:
						# Recurse with the child node as the new parent
						self.growTree(curr_attr)
			# This node has visited all targets
			else:
				# If this solution is better, keep track of its length (number of moves)
				if curr_attr.getT() < self._best_depth:
					self._best_depth = curr_attr.getT()
					print "Current best solution = {} moves".format(self._best_depth)
					# raw_input()

	# Add a node with given attributes to the graph
	def addNode(self, parent_attr, node_attr, action):
		# Create the node object, attach node attributes
		self._graph.add_node(node_attr.getID(), attr=node_attr)

		# Add an edge between the newly created node and its parent
		self.addEdge(parent_attr.getID(), node_attr.getID(), action)

	# Add an edge between a child and parent ID with a given action attribute
	def addEdge(self, parent_id, child_id, action):
		self._graph.add_edge(parent_id, child_id, action=action)

	"""
	Tree utility methods
	"""
	# Returns a dictionary of node_id : attributes (NodeAttributes) for all nodes
	# in the graph
	def getAllNodeAttributes(self):
		return nx.get_node_attributes(self._graph, 'attr')

	# Returns true if the given attribute has position equal to any of its
	# predecessors, false otherwise
	# Essentially, has the agent been in this position before?
	def checkPredecessorCoordinates(self, attr):
		# Get the current node ID
		curr_id = attr.getID()

		# Get the node's position
		a_x, a_y = attr.getPos()

		# Get the dictionary of all node attributes
		all_attr = self.getAllNodeAttributes()

		# Loop until we're at the root node
		while curr_id != 0:
			# Find its parent's ID
			parent_id = self.getPredecessorID(curr_id)

			# Get the parent's position
			t_x, t_y = all_attr[parent_id].getPos()

			# Check whether positions match
			if a_x == t_x and a_y == t_y:
				# A match, the agent has been before
				return True

			# Update node IDs
			curr_id = parent_id

		# No match, the agent hasn't been here before
		return False

	# Get the node ID of the given node's predecessor (parent), ensures that there's
	# only one parent
	def getPredecessorID(self, node_id):
		# Get the list of predecessors for the current node
		predecessor = self._graph.predecessors(node_id)

		# Construct list of predecessor IDs
		pred_id = [i for i in predecessor]

		# Make sure there's only one
		assert(len(pred_id) == 1)

		return pred_id[0]

	"""
	Tree analysis methods
	"""

	# Analyses grid-sizes, max moves without visit and timings
	def analyseParameterTimings(self):
		# Square grid initial size
		grid_init = 2

		# Maximum square grid size
		grid_max = 7

		# Array of grid sizes
		sizes = np.arange(grid_init, grid_max+1)

		# Total grid iterations
		tot_grid = sizes.shape[0]

		# Numpy array to store data to
		data = np.zeros((2, tot_grid))

		# Iterate over grid size
		for i in range(sizes.shape[0]):
			# Current size
			size = sizes[i]

			# Work out max number of targets (but cap it at 5)
			const.NUM_TARGETS = (size*size)-1
			if const.NUM_TARGETS >= 5: const.NUM_TARGETS = 5

			print "Starting episode with square grid of size: {}*{}".format(size, size)

			# Initialise the grid with this size
			self.reset(False, size, size)

			time = self.beginGrowingTree()

			data[0,i] = time
			data[1,i] = size

		print data

		self.visualiseData(data)

	def visualiseData(self, data):
		plt.plot(data[1,:], data[0,:])
		plt.show()

	# Finds solutions with the smallest number of steps (these are the best solutions)
	def findBestSolutions(self):
		# Get the dictionary of node_id : attributes
		node_attr = self.getAllNodeAttributes()

		# Keep a dictionary of solutions
		solution_nodes = dict()

		# Iterate over every node
		for node_id in node_attr:
			# If this node visits all targets (is a solution)
			if node_attr[node_id].getColour() == "green":
				solution_nodes[node_id] = node_attr[node_id].getT()

		# Find the best time
		min_val = min(solution_nodes.itervalues())

		print "Best solution = {} moves".format(min_val)

		# Find keys with min_val
		best_nodes = [k for k, v in solution_nodes.iteritems() if v == min_val]

		# Colour the best solutions differently
		for node_id in best_nodes:
			node_attr[node_id].setColour(n_a)
		nx.set_node_attributes(self._graph, node_attr, 'attr')

		# Construct the best sequence of actions
		actions_list = self.findBestActionSequence(best_nodes)
		print actions_list

		return best_nodes

	# Iterates from a randomly chosen best solution node until it reaches the root node
	# storing the action taken along the way, then reverses that list
	def findBestActionSequence(self, best_nodes):
		# Select a random "best node"
		rand_idx = random.randint(0, len(best_nodes)-1)
		curr_id = best_nodes[rand_idx]

		actions = []

		# Loop until we're at the root node (with ID=0)
		while curr_id != 0:
			# Get the single parent node ID
			pred_id = self.getPredecessorID(curr_id)

			# Get attributes for edge connecting the current and parent node
			edge_attr = self._graph.get_edge_data(pred_id, curr_id)

			# Extract the action for this edge
			action = edge_attr['action']

			# Add edge action at the front of the list (so ordering is correct)
			actions.insert(0, action)

			# Update the node IDs
			curr_id = pred_id

		return actions

	"""
	Visualisation methods
	"""

	def generateColourMap(self):
		node_attr = nx.get_node_attributes(self._graph, 'attr')

		colour_map = [node_attr[node_id].getColour() for node_id in node_attr]

		return colour_map

	def drawNodeLabels(self, pos):
		node_attr = nx.get_node_attributes(self._graph, 'attr')

		node_labels = {node_attr[node_id].getID(): node_attr[node_id].getT() for node_id in node_attr}

		nx.draw_networkx_labels(self._graph, pos, labels=node_labels)

	def drawEdgeLabels(self, pos):
		edge_labels = nx.get_edge_attributes(self._graph, 'action')

		nx.draw_networkx_edge_labels(self._graph, pos, edge_labels=edge_labels)

	# Draw the network and display it
	def visualise(self, visualise=False):
		print "|nodes|={}".format(len(self._graph.nodes()))

		best_nodes = self.findBestSolutions()

		if visualise:
			pos = nx.spring_layout(self._graph)

			colour_map = self.generateColourMap()

			nx.draw(self._graph, pos, node_color=colour_map, with_labels=False)

			self.drawNodeLabels(pos)
			self.drawEdgeLabels(pos)

			plt.show()

	"""
	Utility methods
	"""

	# Given map boundaries, randomly generate a x,y coordinate
	def generateMapPosition(self):
		# Generate a position within bounds
		rand_x = random.randint(0, self._map_w-1)
		rand_y = random.randint(0, self._map_h-1)
		return rand_x, rand_y

	# Given agent starting coordinates, generate target coordinates
	def generateTargets(self, a_x, a_y):
		targets = []

		while len(targets) != const.NUM_TARGETS:
			# Generate a position within bounds
			rand_x, rand_y = self.generateMapPosition()

			ok = True

			# Check agent position
			if rand_x == a_x and rand_y == a_y:
				ok = False
			else:
				# Check other targets
				for pos in targets:
					if rand_x == pos[0] and rand_y == pos[1]:
						ok = False
						break

			if ok: targets.append((rand_x, rand_y))

		return targets

# Attribute container object that each node in the graph/tree contains
class NodeAttributes:
	# Class constructor
	def __init__(	self,
					node_id,
					x,
					y,
					m_w,
					m_h,
					colour=const.DEFAULT_NODE_COLOUR 		):
		"""
		Class attributes
		"""

		# Unique numerical (integer) identifier for this node
		self._id = node_id

		# Position into grid for this node
		self._x = x
		self._y = y

		# Timestep (number of moves made) for this node
		self._t = 0

		# The number of timesteps since this node visited an un-visited target
		self._tsv = 0

		# A binary vector describing which targets have been visited
		self._v = np.zeros(const.NUM_TARGETS)

		# Colour to render this node to
		self._colour = colour

		# Store environment dimensions
		self._map_w = m_w
		self._map_h = m_h

		# Visitation grid (memory of previously visited grid coordinates/positions)
		self._mem = np.zeros((self._map_w, self._map_h))
		self._mem.fill(const.UNVISITED_VAL)
		self._mem[y, x] = const.VISITED_VAL

	"""
	Class Methods
	"""

	# Returns a child instance of the current object with attributes copied across
	# and respective attributes updated to reflect the enacted action/timestep/etc.
	def newUpdatedInstance(self, action, ID):
		# Make a copy of the current instance
		child = self.copy()

		# Update the child's unique identifier
		child.setID(ID)

		# Set child's node colour to red (we can't be the root node here)
		child.setColour(const.DEFAULT_NODE_COLOUR)

		# Enact the given action
		child.enactAction(action)

		# Increment time (distance) and "time since visit" counter
		child.incrementTimeCounters()

		return child

	# Given an action, the position of the agent is updated here
	def enactAction(self, action):
		# Get the agent's current position
		x, y = self.getPos()

		# Enact actions
		if action == 'F': self.setPos(x, y-1)
		if action == 'B': self.setPos(x, y+1)
		if action == 'L': self.setPos(x-1, y)
		if action == 'R': self.setPos(x+1, y)

		# Update the vistation map
		self.updateVisitMap()

	# For a the current agent 2D coordinate, return actions that are possible 
	# (against map boundaries)
	def possibleActions(self):
		# Make a copy of all actions
		actions = list(const.ACTIONS)

		# Get the agent's current position
		x, y = self.getPos()

		# x-dimension
		if x == 0: actions.remove('L')
		elif x == self._map_w - 1: actions.remove('R')

		# y-dimension
		if y == 0: actions.remove('F')
		elif y == self._map_h - 1: actions.remove('B')

		# Further remove actions that are not possible if the agent has already
		# visited particular locations
		# actions = self.possibleActionsForVistationMap(x, y, actions)

		return actions

	# Check whether agent coodinates match with a target position, update the 
	# binary visited vector at the correct index if so
	# Returns a bool if all targets have now been visited
	def checkTargets(self, targets):
		# Quick sanity checks
		assert(len(targets) == self._v.shape[0])
		assert(const.NUM_TARGETS == len(targets))

		# Get the agent's current position
		x, y = self.getPos()

		# Iterate over each target
		for i in range(const.NUM_TARGETS):
			# If this target is unvisited
			if not self._v[i]:
				t_x = targets[i][0]
				t_y = targets[i][1]

				# Check whether the positions match
				if x == t_x and y == t_y:
					# Update visitation vector
					self._v[i] = 1

					# Reset the time since visitation counter
					self.resetTimeSinceVisitation()

					# Check whether we've now visited all targets
					if self.allVisited():
						# Set this node's colour as a solution
						self.setColour(const.SOLUTION_NODE_COLOUR)

						# Signify that all targets have been visited
						return True

		# All targets have not been visited
		return False

	# Given possible actions for boundary cases, remove actions that lead to the 
	# agent visiting a location it has already visited
	def possibleActionsForVistationMap(self, a_x, a_y, actions):
		valid_actions = []

		# Iterate over supplied actions list
		for action in actions:
			if action == 'F' and self._mem[a_y-1, a_x] == const.UNVISITED_VAL:
				valid_actions.append(action)
			elif action == 'B' and self._mem[a_y+1, a_x] == const.UNVISITED_VAL:
				valid_actions.append(action)
			elif action == 'L' and self._mem[a_y, a_x-1] == const.UNVISITED_VAL:
				valid_actions.append(action)
			elif action == 'R' and self._mem[a_y, a_x+1] == const.UNVISITED_VAL:
				valid_actions.append(action)

		return valid_actions

	# Given a list of actions, order them such that the first element is the best action
	# towards visiting the closest unvisited target
	def reorderActions(self, actions, targets):
		ordered_actions = []

		u_d, o_keys = self.orderUnvisitedTargets(targets)

		# Iterate over the ordered list of keys by distance from this agent
		for k in o_keys:
			# Extract the position of the closest unvisited target
			curr_t = u_d[k]

			# Get the best action for this target (may return multiple actions for cases
			# where the relative angle is for example 45 degrees)
			choices = Utility.possibleActionsForAngle(	self.getX(), 
														self.getY(), 
														curr_t[0],
														curr_t[1]		)

			# Make sure something funky hasn't happened (2 for diagonal movement case)
			assert(len(choices) <= 2)

			# Shuffle the list
			random.shuffle(choices)

			# Add choices as long as they're in the list of input possible actions and
			# are NOT a duplicate
			for c in choices:
				if c in actions and c not in ordered_actions:
					ordered_actions.append(c)

		# Append remaining actions to the ordered list
		for action in actions:
			if action not in ordered_actions:
				ordered_actions.append(action)

		# Make sure we're still returning the same number of targets
		assert(len(ordered_actions) == len(actions))

		return ordered_actions

	# Given a list of targets, return an ordered dictionary of unvisited, closest targets
	def orderUnvisitedTargets(self, targets):
		# First construct a list of all unvisited targets
		u = []
		for i in range(len(targets)):
			if not self.getVisited()[i]:
				u.append(targets[i])
		# u = [i for i in range(len(targets)) if not self.getVisited()[i]]

		# Construct dictionary of distance from agent : unvisited target positions
		u_d = {Utility.distanceBetweenPoints(t, self.getPosTuple()) : t for t in u}

		# Ordered list of keys (by distance from agent)
		keylist = sorted(u_d.keys())

		return u_d, keylist

	# Marks the agent's current position in the memory map
	def updateVisitMap(self):
		self._mem[self.getY(), self.getX()] = const.VISITED_VAL

	# Returns true if all targets have been visited, false otherwise
	def allVisited(self):
		if np.sum(self._v) == const.NUM_TARGETS:
			return True
		return False

	# Returns a DEEP copy of this object
	def copy(self):
		return copy.deepcopy(self)

	# Methods to retrieve the name of the current class (NodeAttributes)
	@classmethod
	def getClassname(cls):
		return cls.__name__
	def useClassname(self):
		return self.get_classname()

	"""
	Getters
	"""
	def getID(self):
		return self._id
	def getX(self):
		return self._x
	def getY(self):
		return self._y
	def getPos(self):
		return self._x, self._y
	def getPosTuple(self):
		return (self._x, self._y)
	def getT(self):
		return self._t
	def getVisited(self):
		return self._v
	def getTimeSinceVisit(self):
		return self._tsv
	def getColour(self):
		return self._colour

	"""
	Setters / attribute editors
	"""
	def setID(self, ID):
		self._id = ID
	def setColour(self, colour):
		self._colour = colour
	def setPos(self, x, y):
		self._x = x
		self._y = y
	def incrementTimeCounters(self):
		self._t += 1
		self._tsv += 1
	def resetTimeSinceVisitation(self):
		self._tsv = 0

# Entry method/unit testing
if __name__ == '__main__':
	ts = TreeSolver(	const.MAP_WIDTH, 
						const.MAP_HEIGHT, 
						manual_position=const.MANUAL_SOLVER_POSITIONS	)
	# ts.beginGrowingTree()
	# ts.visualise(visualise=False)
	ts.analyseParameterTimings()
