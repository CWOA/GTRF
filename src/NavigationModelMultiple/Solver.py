#!/usr/bin/env python

import numpy as np
import networkx as nx
import Constants as const
import matplotlib.pyplot as plt

class Solver:
	# Class constructor
	def __init__(	self,
					):
		pass

# Attribute container object that each node contains
class NodeAttributes:
	# Class constructor
	def __init__(	self,
					node_id,
					x,
					y,
					t,
					visited,
					time_since_visit,
					colour='red' 		):
		"""
		Class attributes
		"""

		# Unique numerical (integer) identifier for this node
		self._id = node_id

		# Position into grid for this node
		self._x = x
		self._y = y

		# Timestep (number of moves made) for this node
		self._t = t

		# A binary vector describing which targets have been visited
		self._visited = np.copy(visited)

		# The number of timesteps since this node visited an un-visited target
		self._time_since_visit = time_since_visit

		# Colour to render this node to
		self._colour = colour

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
		return self._visited
	def getTimeSinceVisit(self):
		return self._time_since_visit
	def getColour(self):
		return self._colour

	"""
	Setters
	"""
	def setColour(self, colour):
		self._colour = colour

# Class is a subclass of the "Solver" superclass
class TreeSolver(Solver):
	# Class constructor
	def __init__(	self,
					init_x,
					init_y,
					map_width,
					map_height,
					targets		):
		"""
		Class arguments from init
		"""

		"""
		Class attributes
		"""

		# Initial agent coordinates
		self._init_x = init_x
		self._init_y = init_y

		# Map constraints
		self._map_width = map_width
		self._map_height = map_height

		# Actual directed graph/tree for exploration
		self._graph = nx.Graph()

		# Unique identifier counter for each node
		self._id_ctr = 0

		# Coordinates of targets
		self._targets = targets

		self._max_no_visits = 5

		"""
		Class setup
		"""

		# Create root node attributes
		root_attr = NodeAttributes(		self._id_ctr, 
										self._init_x, 
										self._init_y, 
										0, 
										np.zeros(len(targets)),
										0,
										colour='blue'								)

		# Create root node
		root_id = self.addNode(root_attr)

		# Generate the actual tree
		self.growTree(root_attr)

	def applyActionToPosition(self, action, x, y):
		if action == 'F': return x, y-1
		if action == 'B': return x, y+1
		if action == 'L': return x-1, y
		if action == 'R': return x+1, y

	def addNode(self, node_attr):
		# Unique numerical identifier we're assigning to this node
		node_id = self._id_ctr

		# Create the node object, attach attributes
		self._graph.add_node(node_id, attr=node_attr)

		# Increment the global node ID counter
		self._id_ctr += 1

		return node_id

	def addEdge(self, parent_id, child_id, action):
		self._graph.add_edge(parent_id, child_id, action=action)

	# def checkPositionTargetMatch(self, ):

	def growTree(self, parent_attr):
		# Find possible actions for this current position
		possible_actions = self.possibleActionsForPosition(*parent_attr.getPos())

		# Get some attributes from the parent node
		current_time = parent_attr.getT() + 1
		time_since_visitation = parent_attr.getTimeSinceVisit() + 1

		# Loop over possible actions
		for action in possible_actions:
			# Compute new position based on this action			
			new_x, new_y = self.applyActionToPosition(action, *parent_attr.getPos())

			# Update the store of which targets we've visited
			visited, new_target = self.updateVisitedTargetVector(new_x, new_y, np.copy(parent_attr.getVisited()))

			if new_target: updated_time_since_visit = 0
			else: updated_time_since_visit = time_since_visitation

			colour = 'red'

			if np.sum(visited) == visited.shape[0]:
				colour = 'green'

			node_attr = NodeAttributes(	self._id_ctr,
										new_x,
										new_y,
										current_time,
										visited,
										updated_time_since_visit,
										colour=colour				)

			# Add a new node for this action
			node_id = self.addNode(node_attr)

			# Connect this new node to its parent
			self.addEdge(parent_attr.getID(), node_id, action)

			if np.sum(visited) != visited.shape[0] and time_since_visitation < self._max_no_visits:
				# Recurse with this newly generated node
				self.growTree(node_attr)

	# Check whether supplied coodinates match with a target position, update the 
	# binary visited vector at the correct index if so
	def updateVisitedTargetVector(self, x, y, v):
		# Quick sanity check
		assert(len(self._targets) == v.shape[0])

		# Iterate over each target
		for i in range(len(self._targets)):
			# If this target is unvisited
			if not v[i]:
				t_x = self._targets[i][0]
				t_y = self._targets[i][1]

				# Check whether the positions match
				if x == t_x and y == t_y:
					# print "agent=({},{}), target=({},{})".format(x, y, t_x, t_y)

					# Return the updated visitation vector
					v[i] = 1
					return v, True

		return v, False

	# For a given 2D coordinate, return actions that are possible (against map boundaries)
	def possibleActionsForPosition(self, x, y):
		# Make a copy of all actions
		actions = list(const.ACTIONS)

		# x-dimension
		if x == 0: actions.remove('L')
		elif x == self._map_width - 1: actions.remove('R')

		# y-dimension
		if y == 0: actions.remove('F')
		elif y == self._map_height - 1: actions.remove('B')

		return actions

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

		# Find keys with min_val
		best_nodes = [k for k, v in solution_nodes.iteritems() if v == min_val]

		# Colour the best solutions differently
		for node_id in best_nodes:
			node_attr[node_id].setColour("yellow")
		nx.set_node_attributes(self._graph, node_attr, 'attr')

		return best_nodes

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
	def visualise(self):
		print "|nodes|={}".format(len(self._graph.nodes()))

		best_nodes = self.findBestSolutions()

		pos = nx.spring_layout(self._graph)

		colour_map = self.generateColourMap()

		nx.draw(self._graph, pos, node_color=colour_map, with_labels=False)

		self.drawNodeLabels(pos)
		self.drawEdgeLabels(pos)

		plt.show()

# Entry method/unit testing
if __name__ == '__main__':
	# Initial agent coordinates
	init_x = 0
	init_y = 0

	# Map width/height
	# map_width = const.MAP_WIDTH
	# map_height = const.MAP_HEIGHT
	map_width = 2
	map_height = 2

	# Target coordinates
	# targets = [(0,0), (2,2)]
	targets = [(1,1)]

	ts = TreeSolver(init_x, init_y, map_width, map_height, targets)
	# ts.growTree()
	ts.visualise()
