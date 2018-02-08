#!/usr/bin/env python

import time
import copy
import math
import random
import networkx as nx
import Constants as const
from Utility import Utility
import matplotlib.pyplot as plt

"""
Constructs a globally-optimal solution in a lightweight fashion. Assumes that a global
solution visits all targets, finds the cost of all permutations of target sequences,
returning the best (this is a far better solution than TreeSolver and is essentially
equivalent)
"""

# Class inherits properties of Solver superclass
class SequenceSolver:
	# Class constructor
	def __init__(self):
		"""
		Class attributes/properties
		"""

		self._visualise = False

	"""
	Mandatory class methods
	"""

	def reset(self, agent, targets):
		# Objects for the agent and all targets
		self._agent = agent
		self._targets = targets

		# Number of target objects
		self._num_targets = len(targets)

		# Total number of possible solutions (one or more of which is globally optimal)
		self._complexity = math.factorial(self._num_targets)

		# Reset the graph
		self._graph = nx.DiGraph()

		# Unique identifier counter for each node
		self._id_ctr = 0

	def solve(self):
		self.growTree()
		self._actions = self.findBestSolutions()
		return len(self._actions)

	def nextAction(self):
		return self._actions.pop(0)

	"""
	Tree-growing/utility methods
	"""

	# Initialise root node and begin recursion
	def growTree(self):
		# Create root node
		root = NodeAttributes(	self._id_ctr,
								self._agent,
								self._targets 	)

		# Add the root node
		self._graph.add_node(self._id_ctr, attr=root)

		# Increment the node ID counter
		self._id_ctr += 1

		# Grow the tree recursively
		# tic = time.clock()
		self.growTreeRecursive(root)
		# toc = time.clock()

		# Total time required (seconds)
		# total = toc - tic

		# print "Time to grow tree = {} seconds".format(total)

		if self._visualise:
			self.visualiseTree()

	# Recursively builds or grows the tree until a base case is satisfied
	def growTreeRecursive(self, parent_attr):
		# Find all possible targets to visit
		targets = parent_attr.possibleTargets()

		# Iterate over possible targets to visit
		for target in targets:
			# Create a child attribute for this new target to visit
			curr_attr = parent_attr.newUpdatedInstance(target, self._id_ctr)

			# Increment the node counter
			self._id_ctr += 1

			# Create edge attribute to link this node and its parent
			curr_edge = EdgeAttributes(parent_attr, curr_attr)

			# Add the node and connecting edge to the graph
			self.addNode(parent_attr, curr_attr, curr_edge)

			# Recurse
			self.growTreeRecursive(curr_attr)

	# Add a node for given child, add an edge to its parent
	def addNode(self, parent, child, edge):
		# Create the node
		self._graph.add_node(child.getID(), attr=child)

		# Create the linking edge
		self._graph.add_edge(parent.getID(), child.getID(), attr=edge)

	# Returns a dictionary of node_id : attributes (NodeAttributes) for all nodes
	# in the graph
	def getAllNodeAttributes(self):
		return nx.get_node_attributes(self._graph, 'attr')

	# Returns a dictionary of (parent_id, child_id) : attributes (EdgeAttributes)
	# for all edges in the graph
	def getAllEdgeAttributes(self):
		return nx.get_edge_attributes(self._graph, 'attr')

	# Get the node ID of the given node's predecessor (parent), ensures that there's
	# only one parent
	def getPredecessorID(self, node_id):
		# Get the list of predecessors for the current node
		predecessor = self._graph.predecessors(node_id)

		# Construct list of predecessor IDs
		pred_id = [i for i in predecessor]

		# Make sure there's only one
		assert(len(pred_id) == 1)

		# Return the first item
		return pred_id[0]

	"""
	Tree analysis methods
	"""

	# Finds the solution(s) with the smallest number of steps required to visit all targets
	def findBestSolutions(self):
		# Get all node and edge attributes
		n_a = self.getAllNodeAttributes()
		e_a = self.getAllEdgeAttributes()

		# Solutions dict
		solutions = self.constructSolutionsDict(n_a, e_a)

		# Find the minimal solution
		min_val = min([len(l) for l in solutions.values()])
		
		# Find all node IDs with minimal solutions
		best_nodes = [k for k, v in solutions.iteritems() if len(v) == min_val]

		# Colour the optimal solutions differently
		for node in best_nodes:
			n_a[node].setColour(const.BEST_NODE_COLOUR)
		nx.set_node_attributes(self._graph, n_a, 'attr')

		# Randomly choose the best solution (if there are multiple)
		rand_idx = random.randint(0, len(best_nodes)-1)
		choice = best_nodes[rand_idx]

		# The chosen best solution
		best_solution = solutions[choice]

		return best_solution

	def constructSolutionsDict(self, n_a, e_a):
		# Keep a dictionary of solutions in the form:
		# key: ID of child-most node
		# value: list of ordered actions for this solution
		solutions = dict()

		# Iterate over every node
		for node_id, attr in n_a.iteritems():
			# If this node visits all targets (is a solution and a sink node)
			if attr.getColour() == "green":
				# Get the ID of this node
				curr_id = node_id

				# value: list of actions for this solution
				sol = []

				# Loop until we're at the root node (which must have ID = 0)
				while curr_id != 0:
					# Find this node's parent ID
					parent_id = self.getPredecessorID(curr_id)

					# Find attributes for the edge connecting this node and its parent
					edge_attr = e_a[(parent_id, curr_id)]

					# Prepend the action list
					sol.insert(0, edge_attr.getActionSequence())

					# Old becomes the new
					curr_id = parent_id

				# Flatten the solution list
				flat = [item for sublist in sol for item in sublist]

				# Add this entry to the dictionary
				solutions[node_id] = flat

		return solutions

	"""
	Tree visualisation methods
	"""

	# Render the tree and visualise it
	def visualiseTree(self):
		print "|nodes|={}".format(len(self._graph.nodes()))

		pos = nx.spring_layout(self._graph)

		colour_map = self.generateColourMap()

		nx.draw(self._graph, pos, node_color=colour_map, with_labels=False)

		self.drawNodeLabels(pos)
		self.drawEdgeLabels(pos)

		plt.show()

	def generateColourMap(self):
		node_attr = nx.get_node_attributes(self._graph, 'attr')

		colour_map = [node_attr[node_id].getColour() for node_id in node_attr]

		return colour_map

	def drawNodeLabels(self, pos):
		n_a = nx.get_node_attributes(self._graph, 'attr')

		n_l = {n_a[i].getID(): n_a[i].getObject().getID() for i in n_a}

		nx.draw_networkx_labels(self._graph, pos, labels=n_l)

	def drawEdgeLabels(self, pos):
		e_a = nx.get_edge_attributes(self._graph, 'attr')

		e_l = {e_a[i].getIDTuple(): e_a[i].getActionSequenceLength() for i in e_a}

		nx.draw_networkx_edge_labels(self._graph, pos, edge_labels=e_l)

# Node attribute container object that each node in the graph/tree contains
class NodeAttributes:
	# Class constructor
	def __init__(	self,
					node_id,
					obj,
					targets,
					colour=const.DEFAULT_NODE_COLOUR	):
		"""
		Class attributes/properties
		"""

		# Unique integer identifier for this node
		self._ID = node_id

		# Colour to render this node with
		self._colour = colour

		# This must be the root node if the ID is zero
		if self._ID == 0:
			self._is_root = True
			self._colour = const.ROOT_NODE_COLOUR

		# The object this node represents (could be the agent or a target)
		self._object = obj

		# The list of target IDs that have not been visited by this branch yet
		self._unvisited = targets

	"""
	Class methods
	"""

	def newUpdatedInstance(self, target, ID):
		# Make a deep copy of this instance
		child = self.copy()

		# Update the child's unique ID
		child.setID(ID)

		# Set child's node colour to red (we can't be the root node here)
		child.setColour(const.DEFAULT_NODE_COLOUR)

		# Set the target this new node visits
		child.setObject(target.copy())

		# Remove target we've visited
		child.removeTarget(target)

		# Change the child's node colour if has now visited all targets
		if child.getNumUnvisited() == 0:
			child.setColour(const.SOLUTION_NODE_COLOUR)

		return child

	# Returns a list of possible (UNVISITED) targets to visit
	def possibleTargets(self):
		return self._unvisited

	# Returns a DEEP copy of this object
	def copy(self):
		return copy.deepcopy(self)

	# Given a particular target instance, remove it from the list of unvisited targets
	def removeTarget(self, target):
		to_remove = None

		for t in self._unvisited:
			if t.getID() == target.getID():
				to_remove = t
				break

		if to_remove is not None:
			self._unvisited.remove(to_remove)
		else:
			Utility.die("Trying to remove target that doesn't exist!", __file__)

	def printTargets(self):
		print "Targets: "
		for t in self._unvisited:
			print t

	"""
	Getters
	"""
	def getID(self):
		return self._ID
	def getNumUnvisited(self):
		return len(self._unvisited)
	def getColour(self):
		return self._colour
	def getObject(self):
		return self._object

	"""
	Setters
	"""
	def setID(self, ID):
		self._ID = ID
	def setColour(self, colour):
		self._colour = colour
	def setObject(self, obj):
		self._object = obj

# Edge attribute container object that each edge contains
class EdgeAttributes:
	# Class constructor
	def __init__(	self,
					parent,
					child		):
		"""
		Class attributes/properties
		"""

		# Node identifiers for parent and child this edge connects
		self._parent_ID = parent.getID()
		self._child_ID = child.getID()

		# Coordinates of parent and child within the map
		self._parent_pos = parent.getObject().getPosTuple()
		self._child_pos = child.getObject().getPosTuple()

		# Initialise the list of actions required to get from parent to child nodes
		self._actions = []

		"""
		Class setup
		"""

		# Compute the euclidean distance between the nodes this edge connects
		self.computeDistance()

		# Compute the action sequence between nodes this edge connects
		self.computeActionSequence()

	# Compute the euclidean distance between the nodes this edge connects and set as class
	# attribute
	def computeDistance(self):
		self._distance = Utility.distanceBetweenPoints(self._parent_pos, self._child_pos)

	# Given parent and child node positions within the grid, calculate the shortest action
	# sequence from the parent to the child and store in an ordered list of actions
	def computeActionSequence(self):
		# Get the starting and ending positions
		c_x, c_y = self._parent_pos
		e_x, e_y = self._child_pos

		self._actions = Utility.actionSequenceBetweenCoordinates(c_x, c_y, e_x, e_y)

	"""
	Getters
	"""
	def getIDTuple(self):
		return (self._parent_ID, self._child_ID)
	def getDistance(self):
		return self._distance
	def getActionSequence(self):
		return self._actions
	def getActionSequenceLength(self):
		return len(self._actions)

	"""
	Setters
	"""
	

# Entry method/unit testing
if __name__ == '__main__':
	pass
