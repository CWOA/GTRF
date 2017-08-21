#!/usr/bin/env python

roslib.load_manifest('uav_id')
import rospy as ros

class Coordinator:
	def __init__(self):
		# Need to enable UAV motors via service command

	

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