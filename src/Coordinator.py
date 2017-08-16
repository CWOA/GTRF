#!/usr/bin/env python

roslib.load_manifest('uav_id')
import rospy as ros

# Constant definitions

# Possible UAV states
states = {	
			'GROUNDED': 0, 
			'TAKEOFF': 1, 
			'HOVERING': 2, 
			'MOVE': 3, 
			'EXPERIMENT': 4, 
			'LANDING': 5
		 }

class Coordinator:
	def __init__(self):
		self._current_state = states['GROUNDED']

	def coordinate(self):
		if self._current_state == states['GROUNDED']:

		elif self._current_state == states['TAKEOFF']:

		elif self._current_state == states['HOVERING']:

		elif self._current_state == states['MOVE']:

		elif self._current_state == states['EXPERIMENT']:

		elif self._current_state == states['LANDING']:

		else:
			ros.logerr("State not recognised")

# Entry method
if __name__ == '__main__':
	# Create a coordinator object instance
	c = Coordinator()

	# Initialise this ROS node
	ros.init_node('coordinator')

	# Spin away
	try:
		ros.spin()
	except KeyboardInterrupt:
		print "ImageProcessor node shutting down"