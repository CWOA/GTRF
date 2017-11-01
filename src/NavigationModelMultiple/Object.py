#!/usr/bin/env python

import random as rand
import Constants as const

class Object:
	# Class constructor
	def __init__(	self, 
					is_agent, 
					x=const.AGENT_START_COORDS[0], 
					y=const.AGENT_START_COORDS[1]	):
		### Class attributes/properties

		# Am I the agent or a target?
		self._agent = is_agent

		# Object position (x, y) coordinates
		self._x = x 
		self._y = y

		# Object colour
		if self._agent: self._colour = const.AGENT_COLOUR
		else: const.TARGET_COLOUR

	# Return a tuple of the agent's current coordinates
	def getPos(self):
		return self._x, self._y
		
	def getColour(self):
		return self._colour

class ObjectHandler:
	# Class constructor
	def __init__(	self,
					random_agent_pos=True,
					random_num_targets=False	):
		### Class attributes

		self._agent = None
		self._targets = None

		# Should the agent's position be randomised or some default position
		self._random_agent_pos = random_agent_pos

		# Should we randomise the number of targets or not
		self._random_num_targets = random_num_targets

		# Initialise the class
		self.reset()

	# Reset this handler so we can go again
	def reset(self):
		# Generate a random starting agent coordinate if we're supposed to
		if self._random_agent_pos:
			a_x, a_y = self.generateUnoccupiedPosition()
			self._agent = Object(True, x=a_x, y=a_y)
		# Default agent starting coordinates
		else: self._agent = Object(True)

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
			self._targets.append(Object(False, x=t_x, y=t_y))

	# Generate a random position within the grid that isn't already occupied
	def generateUnoccupiedPosition(self):
		# List of occupied positions
		occupied = []

		# Combine all generated positions up to this point
		if self._agent is not None:
			occupied.append([self._agent.getPos()])
		if self._targets is not None:
			for target in self._targets:
				occupied.append([target.getPos()])

		# Loop until we've generated a valid position
		while True:
			# Generate a position within bounds
			rand_x = rand.randint(0, const.MAP_WIDTH-1)
			rand_y = rand.randint(0, const.MAP_HEIGHT-1)

			ok = True

			# Check the generated position isn't already in use
			for pos in occupied:
				if rand_x == pos[0] and rand_y == pos[1]:
					ok = False

			if ok: return rand_x, rand_y

	# Print coordinates of the agent and all targets
	def printObjectCoordinates(self):
		a_x, a_y = self._agent.getPos()
		print "Agent pos = ({},{})".format(a_x, a_y)

		for i in range(len(self._targets)):
			t_x, t_y = self._targets[i].getPos()
			print "Target #{} pos = ({},{})".format(i, t_x, t_y)

# Entry method for unit testing
if __name__ == '__main__':
	object_handler = ObjectHandler()
	object_handler.printObjectCoordinates()
