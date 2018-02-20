#!/usr/bin/env python

import os
import cv2
import numpy as np
import Constants as const

"""
Class in charge of saving given video to file
"""

class VideoWriter:
	# Class constructor
	def __init__(	self,
					exp_name,
					video_path	):
		"""
		Class arguments
		"""

		# Name of the experiment currently running (used to name files)
		self._exp_name = exp_name

		# Directory to save videos to
		self._video_dir = video_path

		"""
		Class attributes
		"""

		# Codec definition
		self._four_cc = cv2.VideoWriter_fourcc(*const.VIDEO_CODEC)

		# File name counter
		self._file_ctr = 0

	# Open a new video stream
	def reset(self, suffix=None):
		if suffix is None:
			filename = "{}_{}.avi".format(self._exp_name, self._file_ctr)
		else:
			filename ="{}_{}_{}.avi".format(self._exp_name, suffix, self._file_ctr)

		file_path = os.path.join(self._video_dir, filename)

		# OpenCV videowriter object
		self._writer = cv2.VideoWriter(	file_path, 
										self._four_cc, 
										const.VIDEO_FPS, 
										(const.VIDEO_OUT_WIDTH, const.VIDEO_OUT_HEIGHT)	)

		print "Will save video to: {}".format(file_path)

	# Called each time the episode iterates
	def iterate(self, frame):
		# Scale image to output dimensions
		frame = cv2.resize(	frame, 
							(const.VIDEO_OUT_WIDTH, const.VIDEO_OUT_HEIGHT), 
							interpolation=cv2.INTER_NEAREST						)

		# Work out ratio (number of times we need to copy frame to reach desired
		# iteration per second rate given the video framerate)
		ratio = int(const.VIDEO_FPS / const.VIDEO_ITR_PER_SECOND)

		# Write out a frame
		for i in range(ratio):
			self._writer.write(frame)

	# Called when an episode is complete
	def finishUp(self):
		# Close up the writer
		self._writer.release()

		# Increment the file counter
		self._file_ctr += 1

# Entry method for unit testing
if __name__ == '__main__':
	pass
