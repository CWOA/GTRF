#!/usr/bin/env python

import roslib
roslib.load_manifest('uav_id')
import rospy as ros
from gazebo_msgs.msg import *
from std_srvs.srv import Empty

rob_name = "sim_cam"
current_pose = ModelState()

# Position movement vectors [x,y,z]
forward	= []
backward = []
left = []
right = []
up = []
down = []

def updateMovementVectors(gran):
	forward		= [gran,	0,		0]
	backward	= [-gran,	0,		0]
	left		= [0,	 gran,		0]
	right		= [0,	-gran,		0]
	up			= [0,		0,	 gran]
	down		= [0,		0,	-gran]

def move(move_input):
	#enableDisablePhysics(True)
	desired_pose = current_pose
	desired_pose.pose.position.x = current_pose.pose.position.x + move_input[0]
	desired_pose.pose.position.y = current_pose.pose.position.y + move_input[1]
	desired_pose.pose.position.z = current_pose.pose.position.z + move_input[2]
	pos_pub.publish(desired_pose)
	#enableDisablePhysics(False)

def poseCallback(data):
	try:
		idx = data.name.index(rob_name)
		current_pose.model_name = data.name[idx]
		current_pose.pose = data.pose[idx]
		current_pose.twist = data.twist[idx]
		#print current_pose
	except:
		pass

# Entry method
if __name__ == '__main__':
	# Initialise this ROS node
	ros.init_node('grid_camera_controller')

	# Publishers
	pos_pub = ros.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)

	# Subscribers
	pos_sub = ros.Subscriber('/gazebo/model_states', ModelStates, poseCallback)

	# Parameters
	granularity = ros.get_param("~granularity", 1)
	updateMovementVectors(granularity)

	# Main loop
	try:
		while not ros.is_shutdown():
			ros.loginfo("Moving forward")
			move(forward)
			ros.sleep(2)

			ros.loginfo("Moving backward")
			move(backward)
			ros.sleep(2)

			ros.loginfo("Moving up")
			move(up)
			ros.sleep(2)

			ros.loginfo("Moving down")
			move(down)
			ros.sleep(2)

	except ros.ROSInterruptException:
		pass
	finally:
		ros.loginfo("Experiment finished, shutting everything down..")
		ros.signal_shutdown("Experiment finished")
