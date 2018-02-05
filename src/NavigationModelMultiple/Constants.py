#!/usr/bin/env python
# encoding: utf-8

"""
Constants.py
"""

"""
Run-time arguments
"""

# Whether to visualise visual input/map via OpenCV imshow for debugging purposes
VISUALISE = False

# Whether or not to use ROS/Gazebo simulator for synthesised visual input
USE_SIMULATOR = False

# Number of episodes to test on or generate training examples
ITERATIONS = 10000

"""
Algorithm class constants
"""

ALGORITHM_DUAL_INPUT_CNN = 0
ALGORITHM_YOUR_ALGORITHM = 1
# Add your extra algorithm definition here or change the one above

ALGORITHM_METHOD = ALGORITHM_DUAL_INPUT_CNN
# ALGORITHM_METHOD = ALGORITHM_YOUR_ALGORITHM

"""
Directory constats
"""
BASE_DIR = "/home/will/catkin_ws/src/uav_id/tflearn"
DATA_DIR_PICKLE = "data/multiple_nav_data_SIMULATOR.pkl"
DATA_DIR_HDF5 = "data/multiple_nav_data_SIMULATOR.h5"
TENSORBOARD_DIR = "tensorboard"
MODELS_DIR = "models"

"""
ICIP directories
"""
ICIP_DATA_DIR = "ICIP2018/data"
ICIP_FIGURE_DIR = "ICIP2018/figures"
ICIP_MODELS_DIR = "ICIP2018/models"
ICIP_TENSORBOARD_DIR = "ICIP2018/tensorboard"

"""
FieldMap constants
"""
MAP_WIDTH = 10
MAP_HEIGHT = 10
ACTIONS = ['F', 'B', 'L', 'R']

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
Object class consants
"""

# Default agent starting position if random is disabled
AGENT_START_COORDS = (0, 0)

# Spatial distribution method for generating agent/target positions
PRNG_DIST = 0	# Uses "random" python class (which uses Mersenne Twister PRNG)
EQUI_DIST = 1	# Equidistant target spacing
GAUS_DIST = 2	# Gaussian distribution

# OBJECT_DIST_METHOD = PRNG_DIST
# OBJECT_DIST_METHOD = EQUI_DIST
OBJECT_DIST_METHOD = GAUS_DIST

# Unsure whether this should be constant?! Should it somewhat vary per episode?
GAUS_MU_X = 3
GAUS_MU_Y = 5
GAUS_SIGMA_X = 1
GAUS_SIGMA_Y = 1

# Again, unsure whether this should be constant..
EQUI_START_X = 2 # Where equidistant grid starts from
EQUI_START_Y = 3
EQUI_SPACING = 3 # Spacing between equidistant targets

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
MODEL_NAME = "gaussian_SEQUENCE"
DATA_RATIO = 0.9
NUM_EPOCHS = 50
CROSS_VALIDATE = True
NUM_FOLDS = 10

"""
Loop Detector constants
"""
# Use the string-based method (e.g. LRL) or coordinate-based system
USE_ACTION_STRING = False

# Maximum length of loop detection queue at any one time
# MAX_QUEUE_SIZE = 40
MAX_QUEUE_SIZE = 20

# Number of times the agent has to visit a particular coordinate (within)
# the queue before it is deemed to be in an infinite loop
MAX_VISIT_OCCURENCES = 3

"""
Solver constants
"""

SEQUENCE_SOLVER = 0
CLOSEST_SOLVER = 1
TREE_SOLVER = 2
NAIVE_SOLVER = 3

# Which solver to use
# 0: Sequence solver
# 1: Closest solver
# 2: Tree solver
SOLVER_METHOD = SEQUENCE_SOLVER
# SOLVER_METHOD = CLOSEST_SOLVER
# SOLVER_METHOD = NAIVE_SOLVER

"""
Tree Solver constants
"""

# Colour strings for node visualisation in networkx
ROOT_NODE_COLOUR = "blue"
BEST_NODE_COLOUR = "yellow"
SOLUTION_NODE_COLOUR = "green"
DEFAULT_NODE_COLOUR = "red"

# Maximum number of moves since visiting a target before recursion is halted
MAX_TIME_SINCE_VISIT = 10

# Whether Solver.py uses manually defined initial starting positions (for agent, targets)
# useful for keeping episode conditions across episodes and therefore for bug fixing, etc.
MANUAL_SOLVER_POSITIONS = True

"""
Sequence Solver constants
"""

