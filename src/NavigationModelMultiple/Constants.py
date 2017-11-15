#!/usr/bin/env python
# encoding: utf-8

"""
Constants.py
"""

"""
Directory constats
"""
BASE_DIR = "/home/will/catkin_ws/src/uav_id/tflearn"
DATA_DIR_PICKLE = "data/multiple_nav_data_SIMULATOR.pkl"
DATA_DIR_HDF5 = "data/multiple_nav_data_SIMULATOR.h5"
TENSORBOARD_DIR = "tensorboard"
MODELS_DIR = "models"

"""
FieldMap constants
"""
MAP_WIDTH = 10
MAP_HEIGHT = 10
ACTIONS = ['F', 'B', 'L', 'R']
AGENT_START_COORDS = (0, 0)

# Unit to move the agent by each step
MOVE_DIST = 1

# Number of targets to generate if random number of targets is disabled
NUM_TARGETS = 5

# Range of random numbers of targets
NUM_TARGETS_RANGE = (2, 10)

# Method/API to use for saving synthesised training data
# When true, it uses HDF5 (which is suitable for larger databases)
# when false, uses pickle (not sure what it's good for...)
USE_HDF5 = True

"""
VisitationMap constants
"""
UNVISITED_VAL = 0
VISITED_VAL = 1
AGENT_VAL = 10

"""
Visualisation/rendering constants
"""
MAIN_WINDOW_NAME = "Visualisation grid"
AGENT_WINDOW_NAME = "Agent subview"

NUM_CHANNELS = 3

BACKGROUND_COLOUR = (42,42,23)
VISITED_COLOUR = (181,161,62)
AGENT_COLOUR = (89,174,110)
TARGET_COLOUR = (64,30,162)
VISIBLE_COLOUR = (247,242,236)

GRID_PIXELS = 1
WAIT_AMOUNT = 1

"""
SimulatorBridge constants
"""
VIS_ROS_NODE_NAME = "simulator_bridge"

# Topic names
SET_MODEL_STATE_SERVICE_NAME = "/gazebo/set_model_state"
UAV_CAM_IMG_TOPIC_NAME = "/downward_cam/camera/image"

# Downsample ratio for simulated agent subview
IMG_DOWNSAMPLED_WIDTH = 50
IMG_DOWNSAMPLED_HEIGHT = 50

# Default height of agent (this height does NOT vary currently)
DEFAULT_HEIGHT = 3.5

# Gazebo XY plane scale factor (so targets are more distant)
SCALE_FACTOR = 2

# Gazebo name of the agent/robot
ROBOT_NAME = "sim_cam"

# Base name for gazebo cow targets (e.g. cow_0, cow_1, cow_2, ...)
BASE_TARGET_NAME = "cow"

"""
DNN model/training constants
"""
MODEL_NAME = "nav_model_multiple_SIMULATOR"
DATA_RATIO = 0.9
NUM_EPOCHS = 40

"""
Loop Detector constants
"""
# Use the string-based method (e.g. LRL) or coordinate-based system
USE_ACTION_STRING = False

# Maximum length of loop detection queue at any one time
MAX_QUEUE_SIZE = 20

# Number of times the agent has to visit a particular coordinate (within)
# the queue before it is deemed to be in an infinite loop
MAX_VISIT_OCCURENCES = 3
