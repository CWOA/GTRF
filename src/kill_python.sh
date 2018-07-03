# A script for killing all python processes (sometimes necessary when ROS/gazebo doesn't quit gracefully)
ps aux | grep python | grep -v "grep python" | awk '{print $2}' | xargs kill -9
