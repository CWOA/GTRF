#!/usr/bin/env python

import copy
import random
import numpy as np
import networkx as nx
import Constants as const
import matplotlib.pyplot as plt

# Attribute container object that each node contains
class NodeAttributes:
	# Class constructor
	def __init__(	self,
					node_id,
					x,
					y,
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

	# For a the current agent 2D coordinate, return actions that are possible 
	# (against map boundaries)
	def possibleActions(self):
		# Make a copy of all actions
		actions = list(const.ACTIONS)

		# Get the agent's current position
		x, y = self.getPos()

		# x-dimension
		if x == 0: actions.remove('L')
		elif x == const.MAP_WIDTH - 1: actions.remove('R')

		# y-dimension
		if y == 0: actions.remove('F')
		elif y == const.MAP_HEIGHT - 1: actions.remove('B')

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

# Class is a subclass of the "Solver" superclass
class TreeSolver:
	# Class constructor
	def __init__(	self	):
		"""
		Class arguments from init
		"""

		"""
		Class attributes
		"""

		# Initial agent coordinates
		self._init_x, self._init_y = self.generateMapPosition()

		# Randomly generate target positions
		self._targets = self.generateTargets(self._init_x, self._init_y)

		# Actual directed graph/tree for exploration
		self._graph = nx.DiGraph()

		# Unique identifier counter for each node
		self._id_ctr = 0

		"""
		Class setup
		"""

		# Print some stats
		print "Map dimensions = {}*{}".format(const.MAP_WIDTH, const.MAP_HEIGHT)
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
										colour=const.ROOT_NODE_COLOUR	)

		# Create root node within the graph
		self._graph.add_node(self._id_ctr, attr=root_attr)

		# Increment the node ID counter
		self._id_ctr += 1

		# Generate the actual tree
		self.growTree(root_attr)

	# Recursive method for growing the tree/graph
	def growTree(self, parent_attr):
		# Find possible actions for the parent node's current position
		possible_actions = parent_attr.possibleActions()

		# Loop over possible actions
		for action in possible_actions:
			# Create a child attribute with the action enacted
			curr_attr = parent_attr.newUpdatedInstance(action, self._id_ctr)

			# Increment the node counter
			self._id_ctr += 1

			# Create node in the graph
			self.addNode(parent_attr, curr_attr, action)

			# See whether this new position visits unvisited targets
			all_visited = curr_attr.checkTargets(self._targets)

			# See how many timesteps this child has made without visiting a new target
			tsv = curr_attr.getTimeSinceVisit()

			# If this child hasn't visited all targets and hasn't made too many moves
			# without visiting a new target
			if not all_visited and tsv < const.MAX_TIME_SINCE_VISIT:
				# Recurse with the child node as the new parent
				self.growTree(curr_attr)

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
	Tree analysis methods
	"""

	# Returns a dictionary of node_id : attributes (NodeAttributes) for all nodes
	# in the graph
	def getAllNodeAttributes(self):
		return nx.get_node_attributes(self._graph, 'attr')

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
			node_attr[node_id].setColour("yellow")
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
			# Get the list of predecessors for the current node
			predecessor = self._graph.predecessors(curr_id)

			# Construct list of predecessor IDs
			pred_id = [i for i in predecessor]

			# Make sure there's only one
			assert(len(pred_id) == 1)

			# Extract the id
			pred_id = pred_id[0]

			# Get attributes for edge connecting the current and parent node
			edge_attr = self._graph.get_edge_data(pred_id, curr_id)

			# Extract the action for this edge
			action = edge_attr['action']

			# Add edge action at the front of the list (so ordering is correct)
			actions.insert(0, action)

			# Update the node IDs
			curr_id = pred_id

		return actions

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
		rand_x = random.randint(0, const.MAP_WIDTH-1)
		rand_y = random.randint(0, const.MAP_HEIGHT-1)
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

# Entry method/unit testing
if __name__ == '__main__':
	ts = TreeSolver()
	ts.beginGrowingTree()
	ts.visualise(visualise=True)
