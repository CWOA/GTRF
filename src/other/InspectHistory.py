#!/usr/bin/env python

import cv2
import pickle

if __name__ == '__main__':
	pickle_loc = "/home/will/catkin_ws/src/uav_id/output/history_0.pkl"

	with open(pickle_loc, 'rb') as fin:
		history = pickle.load(fin)

	win_name = "Pickle history inspector"

	for h in history:
		cv2.imshow(win_name, h[0])
		print h[1]
		cv2.waitKey(0)