#!/usr/bin/env python

from FieldMap import FieldMap

# Generating training data
def generateTrainingExamples():
	fm = FieldMap(visualise=False, use_simulator=True, save=False)
	fm.startXEpisodes(20000)

# Training model on synthesised data
def trainModel():
	fm = FieldMap(visualise=False, agent_global_view=True, save=True)
	model = dnn_model(fm)
	model.trainModel()

# Testing trained model on real example/problem
def testModel():
	fm = FieldMap(visualise=True)
	fm.startTestingEpisodes(1000)

# Entry method
if __name__ == '__main__':
	generateTrainingExamples()
	# trainModel()
	# testModel()
