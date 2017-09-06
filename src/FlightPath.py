#!/usr/bin/env python

import roslib
roslib.load_manifest('uav_id')
import rospy as ros
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty
from hector_uav_msgs.srv import *
from hector_uav_msgs.msg import *
import actionlib

# Send service command to enable or disable UAV motors
def enableMotors():
	enable_service = "enable_motors"
	ros.wait_for_service(enable_service)
	try:
		enable_motors = ros.ServiceProxy(enable_service, EnableMotors)
		ret = enable_motors(True)
		ros.loginfo("Enabled motors")
	except ros.ServiceException as e:
		ros.logerr("Service call failed for reason: {:s}".format(e))

# Takeoff the UAV (not sure it actually does anything)
def takeOff(client):
	client.send_goal(None)
	client.wait_for_result(ros.Duration.from_sec(5.0))

# Move UAV to a desired pose
def moveToPosition(client, x=0, y=0, z=0):
	pose_stamp = PoseStamped()
	pose_stamp.header.stamp = ros.Time.now()
	pose_stamp.header.frame_id = world_frame
	pose_stamp.pose.position.x = x
	pose_stamp.pose.position.y = y
	pose_stamp.pose.position.z = z
	pose_stamp.pose.orientation.x = 0
	pose_stamp.pose.orientation.y = 0
	pose_stamp.pose.orientation.z = 0
	pose_stamp.pose.orientation.w = 1
	pose = PoseGoal()
	pose.target_pose = pose_stamp

	client.send_goal(pose)


# Entry method
if __name__ == '__main__':
	# Initialise this ROS node
	ros.init_node('flight_path')

	# Publishers
	vel_pub = ros.Publisher('cmd_vel', Twist, queue_size=1)

	# Action clients
	client_takeoff = actionlib.SimpleActionClient('action/takeoff', TakeoffAction)
	client_pose = actionlib.SimpleActionClient('action/pose', PoseAction)
	
	# Parameters - getters
	world_frame = ros.get_param("world_frame")
	# Parameters - setters
	# ros.set_param("/controller/velocity/x/p", 0.1)
	# ros.set_param("/controller/velocity/y/p", 0.1)
	# ros.set_param("/controller/velocity/z/p", 0.1)

	# Ensure servers are enabled/online
	client_takeoff.wait_for_server()
	client_pose.wait_for_server()

	try:
		# Sleeps for x seconds
		ros.sleep(3)

		# Enable the motors
		enableMotors()

		# Sleep for a bit
		ros.sleep(1)

		# Takeoff
		takeOff(client_takeoff)

		ros.sleep(1)

		# Move to a desired pose
		moveToPosition(client_pose, 0, 0, 3)

		ros.sleep(3)

		moveToPosition(client_pose, 5, 0, 3)

		ros.sleep(10)

	except ros.ROSInterruptException:
		pass
	finally:
		ros.loginfo("Experiment finished, shutting everything down..")
		ros.signal_shutdown("Experiment finished")
