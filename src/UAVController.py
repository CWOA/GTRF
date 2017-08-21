#!/usr/bin/env python

roslib.load_manifest('uav_id')
import rospy as ros

# Constant definitions

# Possible UAV states
states = {	
			'GROUNDED': 0,		# Start/Finishing landed/grounded state
			'TAKEOFF': 1,		# UAV is in the process of taking off
			'HOVERING': 2,		# UAV is hovering at a fixed position/attitude
			'MOVE': 3, 			# In process of moving to definted position/attitude
			'EXPERIMENT': 4,	# Control handed over to experiment for training/testing
			'LANDING': 5		# In the process of landing
		 }

class UAVController:
	def __init__(self):
		self._current_state = states['GROUNDED']

		self._motor_service = "enable_motors"

	def fsm(self, msg):
		if self._current_state == states['GROUNDED']:
			
		elif self._current_state == states['TAKEOFF']:

		elif self._current_state == states['HOVERING']:

		elif self._current_state == states['MOVE']:

		elif self._current_state == states['EXPERIMENT']:

		elif self._current_state == states['LANDING']:

		else:
			ros.logerr("State or action not recognised")

	# Send service command to enable or disable UAV motors
	def enableDisableMotors(self, enable):
		ros.wait_for_service(self._motor_service)
		try:
			enable_motors = ros.ServiceProxy(self._motor_service, self._motor_service)
			ret = enable_motors(enable)
		except ros.ServiceException as e:
			ros.logerr("Service call failed: {:s}".format(e))

# Entry method
if __name__ == '__main__':
	# Create a coordinator object instance
	c = UAVController()

	# Initialise this ROS node
	ros.init_node('uav_controller')

	# Spin away
	try:
		ros.spin()
	except KeyboardInterrupt:
		print "ImageProcessor node shutting down"