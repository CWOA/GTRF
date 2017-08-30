#!/usr/bin/env python

roslib.load_manifest('uav_id')
import rospy as ros

# Send service command to enable or disable UAV motors
def enableMotors():
	enable_service = "enable_motors"
	ros.wait_for_service(enable_service)
	try:
		enable_motors = ros.ServiceProxy(enable_service, enable_service)
		ret = enable_motors(True)
	except ros.ServiceException as e:
		ros.logerr("Service call failed for reason: {:s}".format(e))

# Entry method
if __name__ == '__main__':
	pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

	# Initialise this ROS node
	ros.init_node('flight_path')

	rate = rospy.Rate(1) 

	# Spin away
	try:
		enableMotors()

		# Sleeps for 1 second
		ros.sleep(1)

	except rospy.ROSInterruptException:
		pass