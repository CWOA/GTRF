#!/usr/bin/env python

# Core libraries
import sys
sys.path.append('../')
import random
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib as mpl

# My classes
import Constants as const

"""
Class for analysing results, plotting graphs, etc.
"""

class ResultsHelper:
	"""
	Graph drawing utility methods for evaluating algorithm performance
	"""

	@staticmethod
	def drawGenerationTimeGraph(t_s, t_c, s_t, e_t):
		plt.style.use('seaborn-darkgrid')
		# Construct size of targets vector
		T = np.arange(s_t, e_t)

		# Plot vectors
		plt.plot(T, np.average(t_s, axis=1), label="Sequence")
		plt.plot(T, np.average(t_c, axis=1), label="Closest")

		# Graph parameters
		plt.xlabel('|R|')
		plt.ylabel('time(s)')
		plt.legend(loc="center right")
		plt.show()

	@staticmethod
	def drawGenerationLengthGraph(m_s, m_c, s_t, e_t):
		plt.style.use('seaborn-darkgrid')
		# Construct size of targets vector
		T = np.arange(s_t, e_t)

		seq_avg = np.average(m_s, axis=1)
		clo_avg = np.average(m_c, axis=1)

		# print seq_avg
		# print clo_avg

		hist_vec = (m_c - m_s).flatten()

		# print hist_vec

		N, bins, patches = plt.hist(hist_vec, bins=13, normed=True, histtype='stepfilled',)

		# Plot vectors
		# plt.plot(T, seq_avg, label="Sequence")
		# plt.plot(T, clo_avg, label="Closest")

		# Graph parameters
		plt.xlabel('Difference in solution length')
		# plt.ylabel('moves')
		plt.legend(loc="center right")
		plt.tight_layout()
		plt.savefig("{}/solution-generation-hist.pdf".format(Utility.getICIPFigureDir()))
		plt.show()

	@staticmethod
	def drawGenerationGraphs(m_s, m_c, t_s, t_c, num_targets):
		plt.style.use('seaborn-darkgrid')
		fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True)

		# Number of targets array
		R = np.asarray(num_targets)

		# Average results
		m_seq_avg = np.average(m_s, axis=1)
		m_clo_avg = np.average(m_c, axis=1)
		t_seq_avg = np.average(t_s, axis=1)
		t_clo_avg = np.average(t_c, axis=1)

		axs[0].plot(R, m_seq_avg, 'b', label="Target Ordering")
		axs[0].plot(R, m_clo_avg, 'r', label="Closest Unvisited")
		axs[1].plot(R, t_seq_avg, 'b', label="Target Ordering")
		axs[1].plot(R, t_clo_avg, 'r', label="Closest Unvisited")

		axs[0].set_ylabel("|Moves|")
		axs[0].set_xlabel("|R|")
		axs[1].set_ylabel("Time (s)")
		axs[1].set_xlabel("|R|")

		plt.legend(loc="upper left")
		plt.tight_layout()
		plt.savefig("{}/solution-generation.pdf".format(Utility.getICIPFigureDir()))
		plt.show()

	# Draws graph of training data instance size versus best validation accuracy
	@staticmethod
	def drawDatasetSizeAccuracyGraph():
		# Load the data
		base = Utility.getICIPDataDir()
		val_5k = np.genfromtxt("{}/5k_val_acc.csv".format(base), delimiter=',', skip_header=1, names=['x', 'y', 'z'])
		val_10k = np.genfromtxt("{}/10k_val_acc.csv".format(base), delimiter=',', skip_header=1, names=['x', 'y', 'z'])
		val_20k = np.genfromtxt("{}/20k_val_acc.csv".format(base), delimiter=',', skip_header=1, names=['x', 'y', 'z'])
		val_40k = np.genfromtxt("{}/40k_val_acc.csv".format(base), delimiter=',', skip_header=1, names=['x', 'y', 'z'])
		val_60k = np.genfromtxt("{}/60k_val_acc.csv".format(base), delimiter=',', skip_header=1, names=['x', 'y', 'z'])

		# Extract maximum validation values
		max_5 = val_5k['z'].max()
		max_10 = val_10k['z'].max()
		max_20 = val_20k['z'].max()
		max_40 = val_40k['z'].max()
		max_60 = val_60k['z'].max()

		x = np.asarray([5000, 10000, 20000, 40000, 60000])
		y = np.asarray([max_5, max_10, max_20, max_40, max_60])

		plt.style.use('seaborn-darkgrid')
		plt.plot(x, y)
		plt.xlabel('Dataset size')
		plt.ylabel('Accuracy')
		plt.tight_layout()
		plt.savefig("{}/dataset-size-accuracy.pdf".format(Utility.getICIPFigureDir()))
		plt.show()

	# Method for drawing a graph that compares training and validation accuracy versus
	# epochs.
	@staticmethod
	def drawAccuracyGraph():
		# Convolutional based smoothing function
		def smooth(y, box_pts):
		    return savgol_filter(y, box_pts, 3)
		    # return y_smooth

		# Load the data
		base = Utility.getICIPDataDir()
		acc_5k = np.genfromtxt("{}/5k_train_acc.csv".format(base), delimiter=',', skip_header=1, names=['x', 'y', 'z'])
		val_5k = np.genfromtxt("{}/5k_val_acc.csv".format(base), delimiter=',', skip_header=1, names=['x', 'y', 'z'])
		acc_10k = np.genfromtxt("{}/10k_train_acc.csv".format(base), delimiter=',', skip_header=1, names=['x', 'y', 'z'])
		val_10k = np.genfromtxt("{}/10k_val_acc.csv".format(base), delimiter=',', skip_header=1, names=['x', 'y', 'z'])
		acc_20k = np.genfromtxt("{}/20k_train_acc.csv".format(base), delimiter=',', skip_header=1, names=['x', 'y', 'z'])
		val_20k = np.genfromtxt("{}/20k_val_acc.csv".format(base), delimiter=',', skip_header=1, names=['x', 'y', 'z'])
		acc_40k = np.genfromtxt("{}/40k_train_acc.csv".format(base), delimiter=',', skip_header=1, names=['x', 'y', 'z'])
		val_40k = np.genfromtxt("{}/40k_val_acc.csv".format(base), delimiter=',', skip_header=1, names=['x', 'y', 'z'])
		acc_60k = np.genfromtxt("{}/60k_train_acc.csv".format(base), delimiter=',', skip_header=1, names=['x', 'y', 'z'])
		val_60k = np.genfromtxt("{}/60k_val_acc.csv".format(base), delimiter=',', skip_header=1, names=['x', 'y', 'z'])

		# Define the style and subplots
		plt.style.use('seaborn-darkgrid')
		fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

		# Scale x-axis values down to the number of epochs
		acc_5k['y'] = (const.NUM_EPOCHS*(acc_5k['y'] - acc_5k['y'].min())) / acc_5k['y'].max()
		acc_10k['y'] = (const.NUM_EPOCHS*(acc_10k['y'] - acc_10k['y'].min())) / acc_10k['y'].max()
		acc_20k['y'] = (const.NUM_EPOCHS*(acc_20k['y'] - acc_20k['y'].min())) / acc_20k['y'].max()
		acc_40k['y'] = (const.NUM_EPOCHS*(acc_40k['y'] - acc_40k['y'].min())) / acc_40k['y'].max()
		acc_60k['y'] = (const.NUM_EPOCHS*(acc_60k['y'] - acc_60k['y'].min())) / acc_60k['y'].max()

		val_5k['y'] = ((const.NUM_EPOCHS*(val_5k['y'] - val_5k['y'].min())) / val_5k['y'].max()) + 1
		val_10k['y'] = ((const.NUM_EPOCHS*(val_10k['y'] - val_10k['y'].min())) / val_10k['y'].max()) + 1
		val_20k['y'] = ((const.NUM_EPOCHS*(val_20k['y'] - val_20k['y'].min())) / val_20k['y'].max()) + 1
		val_40k['y'] = ((const.NUM_EPOCHS*(val_40k['y'] - val_40k['y'].min())) / val_40k['y'].max()) + 1 
		val_60k['y'] = ((const.NUM_EPOCHS*(val_60k['y'] - val_60k['y'].min())) / val_60k['y'].max()) + 1

		# Alpha constant
		alpha = 0.2

		# Smoothing parameter
		tra_s = 99

		# Plot to training accuracy graph
		axs[0].plot(acc_5k['y'], acc_5k['z'], color='y', alpha=alpha)
		axs[0].plot(acc_10k['y'], acc_10k['z'], color='k', alpha=alpha)
		axs[0].plot(acc_20k['y'], acc_20k['z'], color='r', alpha=alpha)
		axs[0].plot(acc_40k['y'], acc_40k['z'], color='g', alpha=alpha)
		axs[0].plot(acc_60k['y'], acc_60k['z'], color='b', alpha=alpha)

		axs[0].plot(acc_5k['y'], smooth(acc_5k['z'], tra_s), color='y', label='5k')
		axs[0].plot(acc_10k['y'], smooth(acc_10k['z'], tra_s), color='k', label='10k')
		axs[0].plot(acc_20k['y'], smooth(acc_20k['z'], tra_s), color='r', label='20k')
		axs[0].plot(acc_40k['y'], smooth(acc_40k['z'], tra_s), color='g', label='40k')
		axs[0].plot(acc_60k['y'], smooth(acc_60k['z'], tra_s), color='b', label='60k')

		# Plot to validation accuracy graph
		axs[1].plot(val_5k['y'], val_5k['z'], color='y', alpha=alpha)
		axs[1].plot(val_10k['y'], val_10k['z'], color='k', alpha=alpha)
		axs[1].plot(val_20k['y'], val_20k['z'], color='r', alpha=alpha)
		axs[1].plot(val_40k['y'], val_40k['z'], color='g', alpha=alpha)
		axs[1].plot(val_60k['y'], val_60k['z'], color='b', alpha=alpha)

		# Smoothing constant
		val_s = 11

		axs[1].plot(val_5k['y'], smooth(val_5k['z'], val_s), color='y', label='5k')
		axs[1].plot(val_10k['y'], smooth(val_10k['z'], val_s), color='k', label='10k')
		axs[1].plot(val_20k['y'], smooth(val_20k['z'], val_s), color='r', label='20k')
		axs[1].plot(val_40k['y'], smooth(val_40k['z'], val_s), color='g', label='40k')
		axs[1].plot(val_60k['y'], smooth(val_60k['z'], val_s), color='b', label='60k')

		# Set axis labels for subplots
		axs[0].set_xlabel("Epochs")
		axs[0].set_ylabel("Accuracy")
		axs[0].set_title("Training")
		axs[1].set_xlabel("Epochs")
		axs[1].set_title("Validation")

		plt.axis([0, 50, 0.5, 0.9])
		plt.legend(loc="upper right")
		plt.tight_layout()

		plt.savefig("{}/motion-training-accuracy.pdf".format(Utility.getICIPFigureDir()))
		plt.show()

	# List results from a given filepath to a numpy results file
	@staticmethod
	def listResults(results_filepath):
		print "Results for file {}".format(results_filepath)

		data = np.load(results_filepath)

		# Retrieve the number of instances
		num_inst = data.shape[0]

		# Numpy array for testing data, consists of:
		# 0: number of moves required by the model
		# 1: number of moves required by employed solver (closest, target ordering)
		# 2: number of times loop detection is triggered
		# 3: Average discovery/per timestep

		# % of times when a loop was detected
		num_loops = np.where(data[:,2] > 0)[0].shape[0]
		print "__Loop detected________________"
		print "{}%\n".format((float(num_loops)/num_inst)*100)

		# Find stats about how many times loop detection is triggered again once it 
		# has already been triggered
		mult_loops = np.where(data[:,2] > 1)[0].shape[0]
		print "__Multiple Loops detected________________"
		if num_loops > 0:
			print "{}%\n".format((float(mult_loops)/num_loops)*100)
		else:
			print "0%"

		# Find stats about % of time model generates over 100 moves
		over_100 = np.where(data[:,0] > 100)[0].shape[0]
		print "__Over 100 moves________________"
		print "{}%\n".format((float(over_100)/num_inst)*100)

		hist_vec = data[:,0] - data[:,1]

		# Find stats about % of time model generates globally-optimal solution
		optimal = np.where(hist_vec == 0)[0].shape[0]
		print "__Globally Optimal?________________"
		print "{}%\n".format((float(optimal)/num_inst)*100)

		# Find stats about % of time model generates <10 difference than globally-optimal solution
		diff_10 = np.where(hist_vec < 10)[0].shape[0]
		print "__<10 Difference________________"
		print "{}%\n".format((float(diff_10)/num_inst)*100)

		# Find the target discovery rate and respective standard deviation
		avg_dt = np.mean(data[:,3])
		sig_dt = np.std(data[:,3])
		print "__Discovery rate________________"
		print "{}+/-{}".format(avg_dt, sig_dt)

	@staticmethod
	def drawModelLengthHistogram():
		base = Utility.getICIPDataDir()
		data_seq = np.load("{}/test_data_seq.npy".format(base))
		data_clo = np.load("{}/test_data_clo.npy".format(base))
		data_nav = np.load("{}/test_data_NAIVE.npy".format(base))
		data_seq_sim = np.load("{}/test_data_SIMULATOR_seq.npy".format(base))
		#data_seq_sim = np.load("{}/test_data_GAUS_SEQ.npy".format(base))

		assert(data_seq.shape == data_clo.shape)
		assert(data_clo.shape == data_nav.shape)
		assert(data_seq_sim.shape == data_nav.shape)

		num_inst = data_seq_sim.shape[0]

		# Find stats about how many times loop detection was triggered
		loop_seq = np.where(data_seq[:,2] > 0)[0].shape[0]
		loop_clo = np.where(data_clo[:,2] > 0)[0].shape[0]
		loop_seq_sim = np.where(data_seq_sim[:,2] > 0)[0].shape[0]
		print "__Loop detected________________"
		print "TO: {}%".format((float(loop_seq)/num_inst)*100)
		print "CU: {}%".format((float(loop_clo)/num_inst)*100)
		print "TO+S: {}%".format((float(loop_seq_sim)/num_inst)*100)
		print "\n\n"

		# Find stats about how many times loop detection is triggered again once it 
		# has already been triggered
		doub_seq = np.where(data_seq[:,2] > 1)[0].shape[0]
		doub_clo = np.where(data_clo[:,2] > 1)[0].shape[0]
		doub_seq_sim = np.where(data_seq_sim[:,2] > 1)[0].shape[0]
		print "__Multiple Loops detected________________"
		print "TO: {}%".format((float(doub_seq)/loop_seq)*100)
		print "CU: {}%".format((float(doub_clo)/loop_clo)*100)
		print "TO+S: {}%".format((float(doub_seq_sim)/loop_seq_sim)*100)
		print "\n"

		# Find stats about % of time model generates over 100 moves
		over_seq = np.where(data_seq[:,0] > 100)[0].shape[0]
		over_clo = np.where(data_clo[:,0] > 100)[0].shape[0]
		over_nav = np.where(data_nav[:,0] > 100)[0].shape[0]
		over_seq_sim = np.where(data_seq_sim[:,0] > 100)[0].shape[0]
		print "__Over 100 moves________________"
		print "TO: {}%".format((float(over_seq)/num_inst)*100)
		print "CU: {}%".format((float(over_clo)/num_inst)*100)
		print "NS: {}%".format((float(over_nav)/num_inst)*100)
		print "TO+S: {}%".format((float(over_seq_sim)/num_inst)*100)
		print "\n"

		hist_vec_seq = data_seq[:,0] - data_seq[:,1]
		hist_vec_clo = data_clo[:,0] - data_clo[:,1]
		hist_vec_nav = data_nav[:,0] - data_nav[:,1]
		hist_vec_seq_sim = data_seq_sim[:,0] - data_seq_sim[:,1]

		# Find stats about % of time model generates globally-optimal solution
		opt_seq = np.where(hist_vec_seq == 0)[0].shape[0]
		opt_clo = np.where(hist_vec_clo == 0)[0].shape[0]
		opt_nav = np.where(hist_vec_nav == 0)[0].shape[0]
		opt_seq_sim = np.where(hist_vec_seq_sim == 0)[0].shape[0]
		print "__Globally Optimal?________________"
		print "TO: {}%".format((float(opt_seq)/num_inst)*100)
		print "CU: {}%".format((float(opt_clo)/num_inst)*100)
		print "NS: {}%".format((float(opt_nav)/num_inst)*100)
		print "TO+S: {}%".format((float(opt_seq_sim)/num_inst)*100)
		print "\n"

		# Find stats about % of time model generates <10 difference than globally-optimal solution
		dif_seq = np.where(hist_vec_seq < 10)[0].shape[0]
		dif_clo = np.where(hist_vec_clo < 10)[0].shape[0]
		dif_nav = np.where(hist_vec_nav < 10)[0].shape[0]
		dif_seq_sim = np.where(hist_vec_seq_sim < 10)[0].shape[0]
		print "__<10 Difference________________"
		print "TO: {}%".format((float(dif_seq)/num_inst)*100)
		print "CU: {}%".format((float(dif_clo)/num_inst)*100)
		print "NS: {}%".format((float(dif_nav)/num_inst)*100)
		print "TO+S: {}%".format((float(dif_seq_sim)/num_inst)*100)
		print "\n"

		hist_vec = np.zeros((data_seq.shape[0], 4))

		hist_vec[:,0] = data_seq[:,0] - data_seq[:,1]
		hist_vec[:,1] = data_clo[:,0] - data_clo[:,1]
		hist_vec[:,2] = data_nav[:,0] - data_nav[:,1]
		hist_vec[:,3] = data_seq_sim[:,0] - data_seq_sim[:,1]

		plt.style.use('seaborn-darkgrid')

		plt.hist(	hist_vec, bins=80, normed=True, histtype='step',
					color=['g', 'b', 'k', 'r'],
					label=['TO', 'CU', 'NS', 'TO+S'], stacked=False		)

		plt.xlabel("Difference in solution length")
		plt.ylabel("Probability")
		plt.axis([0, 200, 0, 0.045])
		plt.legend()
		plt.tight_layout()

		plt.savefig("{}/model-solution-length.pdf".format(Utility.getICIPFigureDir()))
		plt.show()

# Entry method/unit testing
if __name__ == '__main__':
	# ResultsHelper.drawDatasetSizeAccuracyGraph()
	# ResultsHelper.drawAccuracyGraph()

	ResultsHelper.listResults("/home/will/catkin_ws/src/uav_id/tflearn/ICIP2018/data/RESULTS_naive_solution.npy")
	# Utility.drawModelLengthHistogram()
