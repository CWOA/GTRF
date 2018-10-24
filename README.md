# GTRF (Grid-based Target Recovery Framework)

This repository provides a framework for rapidly prototyping, verifying and evaluating algorithmic approaches to the problem of distributed static and moving target recovery when given limited environment visibility.

This repository accompanies the paper to be published as part of the 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) proceedings entitled *"Deep Learning for Exploration and Recovery of Uncharted and Dynamic Targets from UAV-like Vision"*.
Therefore, for implementation details, results, etc. see the published paper.
All source code used to generate results and data within the paper is given here for public use, further work, replicability and transparency.
The video accompanying the paper -- providing an overview of the problem definition itself, the implemented dual-stream paradigm as well as performance results across a variety of experiments -- can be watched at: https://vimeo.com/280747562

Installation
------
1. Simply clone this repository: *https://github.com/CWOA/gtrf* (if using ROS, place repository in your catkin workspace)
2. Install the dependencies below as necessary

Essential dependencies
------
  * OpenCV (https://opencv.org/) - used for rendering visual input to the respective algorithm
  * Numpy (http://www.numpy.org/) - used for numerical/matrix operations
  * NetworkX (https://networkx.github.io/) - used for compute globally-optimal solutions to epsiode generations

Non-essential dependencies
------
  * h5py (https://pypi.org/project/h5py/) - for storing large synthesised datasets to file.
  * TQDM (https://pypi.python.org/pypi/tqdm) - progressbar visualisation package for monitoring iterative progress in large operations (e.g. training data generation, evaluation)
  * Tensorflow v1.0+ (https://www.tensorflow.org/) - installed to support TFLearn
  * TFLearn (http://tflearn.org/) - high-level API for Tensorflow DNN operations
  * Matplotlib (https://matplotlib.org/) - used for plotting graph data
  * Scikit-learn (http://scikit-learn.org/stable/) - used for randomly segregating training towards cross-fold validation
  
ROS dependencies
------
If using ROS; you want to operate on realistic visual input from the simulator, the following dependencies must be installed: 
  * ROS (http://www.ros.org/) - used to interact between main classes and Gazebo and simulate UAV-like properties
  * Gazebo (http://gazebosim.org/) - used to render simulated camera view for UAV downward vision
  * Hector Quadrotor (http://wiki.ros.org/hector_quadrotor) - used to simulate the UAV agent in simulation within Gazebo

Getting started
------
* Run the file "src/NavigationModelMultiple.py"
* To add your own algorithm, edit the file "YourAlgorithm.py"
* Experiment configurations, constants, etc. can be found in "src/Constants.py"
* To run via ROS and Gazebo, run the command "roslaunch gtrf gtrf.launch" which spawns a simulated UAV, the environment itself (containing 5 target cow models) and runs the file "src/NavigationModelMultiple.py".
* Edit the base directory constant (BASE_DIR) in "src/Constants.py" to point at the complete path at which you want to store training data, model weights, etc. (e.g. "/home/USERNAME/catkin_ws/src/gtrf/data")

License
------
Non-Commercial Government License (see [LICENSE](LICENSE))
