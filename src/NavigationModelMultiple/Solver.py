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
					visited			):
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
		self._time_since_visit

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
		self._graph = nx.DiGraph()

		# Unique identifier counter for each node
		self._id_ctr = 0

		# Coordinates of targets
		self._targets = targets

		self._max_no_visits = 10

		"""
		Class setup
		"""

		# Create root node attributes
		root_attr = NodeAttributes(		self._id_ctr, 
										self._init_x, 
										self._init_y, 
										0, 
										np.zeros(len(targets))		)

		# Create root node
		root_id = self.addNode(			root_attr.getX(),
										root_attr.getY(),
										root_attr.getT(),
										root_attr.getVisited()		)

		# Generate the actual tree
		self.growTree(root_attr)

	def applyActionToPosition(self, action, x, y):
		if action == 'F': return x, y-1
		if action == 'B': return x, y+1
		if action == 'L': return x-1, y
		if action == 'R': return x+1, y

	def addNode(self, x, y, t, v):
		# Unique numerical identifier we're assigning to this node
		node_id = self._id_ctr

		# Create a node attribute object with the relevant data
		attributes = NodeAttributes(node_id, x, y, t, v)

		# Create the node object, attach attributes
		self._graph.add_node(node_id, attr=attributes)

		# Increment the global node ID counter
		self._id_ctr += 1

		return node_id

	def addEdge(self, parent_id, child_id, action):
		self._graph.add_edge(parent_id, child_id, action=action)

	# def checkPositionTargetMatch(self, ):

	def growTree(self, parent_attr):
		# Find possible actions for this current position
		possible_actions = self.possibleActionsForPosition(*parent_attr.getPos())

		current_time = parent_attr.getT() + 1

		# Loop over possible actions
		for action in possible_actions:
			# Compute new position based on this action			
			new_x, new_y = self.applyActionToPosition(action, *parent_attr.getPos())

			# Update the store of which targets we've visited
			visited = self.updateVisitedTargetVector(new_x, new_y, parent_attr.getVisited())

			# Add a new node for this action
			node_id = self.addNode(new_x, new_y, current_time, visited)

			# Connect this new node to its parent
			self.addEdge(parent_attr.getID(), node_id, action)

			if np.sum(visited) != visited.shape[0] and time_since_visitation < self._max_no_visits:
				# Recurse with this newly generated node
				self.growTree(NodeAttributes(node_id, new_x, new_y, current_time, visited))

	# Check whether supplied coodinates match with a target position, update the 
	# binary visited vector at the correct index if so
	def updateVisitedTargetVector(self, x, y, v):
		# Quick sanity check
		assert(len(self._targets) == v.shape[0])

		# Iterate over each target
		for i in range(len(self._targets)):
			# If this target is unvisited
			if not v[i]:
				# Check whether the positions match
				if x == self._targets[i][0] and y == self._targets[i][1]:
					# Return the updated visitation vector
					v[i] = 1
					break

		return v

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

	# Draw the network and display it
	def visualise(self):
		nx.draw(self._graph)
		plt.show()

# Entry method/unit testing
if __name__ == '__main__':
	# Initial agent coordinates
	init_x = 1
	init_y = 1

	# Map width/height
	# map_width = const.MAP_WIDTH
	# map_height = const.MAP_HEIGHT
	map_width = 3
	map_height = 3

	# Target coordinates
	targets = [(0,0), (2,2)]

	ts = TreeSolver(init_x, init_y, map_width, map_height, targets)
	ts.growTree()
	ts.visualise()
