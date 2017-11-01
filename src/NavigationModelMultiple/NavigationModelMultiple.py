#!/usr/bin/env python

import os

def testModelOnRealExample(self):
	print "Testing model on real examples"

	# Load the model from file
	self.loadModel()

	# Sanity check
	# X0_train, X0_test, X1_train, X1_test, Y_train, Y_test = self.loadData()
	# self.evaluateModel(X0_test, X1_test, Y_test)

	# Object to detect infinite agent loops
	detector = LoopDetector()

	# Number of test/examples to run in total
	num_examples = 1000

	upper_num_moves = self._FM._grid_height * self._FM._grid_width
	num_under = 0

	for i in range(num_examples):
		# Reset the grid
		self._FM.reset()

		# Reset the detector
		detector.reset()

		# Number of targets the agent has visited
		num_visited = 0

		# Number of moves the agent has made
		num_moves = 0

		# Indicator of whether the agent is stuck
		agent_stuck = False

		# Render the updated view
		render_img, subview = self._FM.render()

		# Loop until we've visited all the targets
		while num_visited != self._FM._num_targets:
			# Get a copy of the visitation map
			visit_map = self._FM._map.copy()

			# Mark the current location of the agent
			visit_map[self._FM._agent_y, self._FM._agent_x] = 10

			# Based on this state, use the trained model to predict where to go
			prediction = self.testModelSingle(subview, visit_map)

			# Find the index of the max argument
			max_idx = np.argmax(prediction)
			choice = np.zeros(self._num_classes)
			choice[max_idx] = 1
			action = self._FM.classVectorToAction(choice)

			# Add the suggested action and check history, check if the agent is
			# stuck in a loop, act accordingly
			if not agent_stuck and detector.addCheckAction(action):
				agent_stuck = True
				# print "Detected infinite agent loop."

			# Agent is stuck, move towards nearest unvisited location
			if agent_stuck:
				action = self._FM.findUnvisitedDirection()
				# print "Agent in unstucking mode, moving: {}".format(action)
				# cv2.waitKey(0)

			# Make the move (Forward, Backward, Left, Right)
			has_visited, reward = self._FM.performAction(action)
			num_visited += reward

			# Check whether the agent is still stuck following a performed action
			if agent_stuck and not has_visited:
				agent_stuck = False
				# print "Agent is no longer stuck!"

			# Increment the number of moves made by the agent
			num_moves += 1

			# Render the updated view
			render_img, subview = self._FM.render()

			# Display the images
			cv2.imshow(self._FM._window_name, render_img)
			cv2.imshow(self._FM._window_name_agent, subview)
			# print action
			# print visit_map
			cv2.waitKey(1)

		# Print some stats
		print "Solving example {}/{} took {} moves".format(i+1, num_examples, num_moves)

		# If the agent made under 100 moves, record so
		if num_moves < upper_num_moves:
			num_under += 1

	# Print some more stats
	percent_correct = float(num_under/num_examples) * 100
	print "{}/{} under {} moves, or {}% success".format(	num_under,
															num_examples,
															upper_num_moves,
															percent_correct		)



# Entry method
if __name__ == '__main__':
	### Generating training data
	# fm = FieldMap(visualise=False, agent_global_view=True, save=True)
	# fm.startXEpisodes(20000)

	### Training model on synthesised data
	# fm = FieldMap(visualise=False, agent_global_view=True, save=True)
	# model = dnn_model(fm)
	# model.trainModel()

	### Testing trained model on real example/problem
	fm = FieldMap(visualise=True)
	model = dnn_model(fm)
	model.testModelOnRealExample()
