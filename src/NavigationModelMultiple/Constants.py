#!/usr/bin/env python
# encoding: utf-8

"""
Constants.py
"""

"""
Directory constats
"""
BASE_DIR = "/home/will/catkin_ws/src/uav_id/tflearn"
DATA_DIR = "data/multiple_nav_data.pkl"
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

# Default height of agent (this height does NOT vary)
DEFAULT_HEIGHT = 3.5

SCALE_FACTOR = 2

# Gazebo name of the agent/robot
ROBOT_NAME = "sim_cam"

# Base name for gazebo cow targets (e.g. cow_0, cow_1, cow_2, ...)
BASE_TARGET_NAME = "cow"

"""
DNN model/training constants
"""
MODEL_NAME = "visit_map_navigation_model"
DATA_RATIO = 0.9
NUM_EPOCHS = 40