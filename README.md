# Overview

LLM-based robot control experimentation

The ros2_ai_agent directory contains two experimentation AI agents for robotics:

   - ros2_ai_agent_turtlesim
   - ros2_ai_agent_robotic_arm



## Installation

The repository can be cloned as follows:

   - git clone https://github.com/AlbertaBeef/llm-robot-control
   - cd llm-robot-control
   
## Installing Dependencies

The following dependencies are required:

   - langchain
   - langchain-core
   - langchain-community
   - langchain-openai
   - langchain-ollama


The robotic arm agent requires the ur_simulation_gz package, from Universal Robots, which can be installed as follows:

   - apt install ros-jazzy-ur
   - apt install ros-jazzy-ur-simulation-gz
   
   
# Building the ROS2 AI Agent package

The ros2_ai_agent package can be built and installed as follows:

   - cd ros2_ai_agent
   - rosdep update && rosdep install --ignore-src --from-paths . -y
   - colcon build
   - source install/setup.bash


# Prior to launch the ROS2 AI Agent demos

If using OLLAMA to run LLMs locally, make sure the server is running (automatically, or explicitly as follows):

   - ollama serve 
   
If using OpenAI to run LLMs in the cloud, define your OPENAI_API_KEY.



# Lauching the ROS2 AI Agent for Turtlesim

If using OLLAMA to run LLMs locally, make sure the server is running (ie. ollama serve)
If using OpenAI to run LLMs in the cloud, define your OPENAI_API_KEY.

This demo can be run using two terminals.

In the first terminal, launch the ROS2 AI Agent, and Turtlesim, as follows:

   - ros2 launch ros2_ai_agent start_turtlesim_agent.launch.py
  
By default, all the tools are enabled.  The agent is configuration with the following launch arguments:

   - use_basic_tools:=True|False : get_ros_distro, get_domain_id
   - use_generic_tools:=True|False : list_topics, list_nodes, list_services, list_actions
   - use_robot_tools:=True|False : move_forward, rotate, get_pose
    
In the second terminal, send user prompts to the ROS2 AI Agent as strings to the /llm_prompt topic:

   - ros2 topic pub -1 /llm_prompt std_msgs/msg/String "{data: 'List the ROS topics.'}"
   - ros2 topic pub -1 /llm_prompt std_msgs/msg/String "{data: 'Get the current position of the turtle.'}"
   - ros2 topic pub -1 /llm_prompt std_msgs/msg/String "{data: 'Advance the turtle 10.0 units.'}"
   - ros2 topic pub -1 /llm_prompt std_msgs/msg/String "{data: 'Turn the turtle 90 degres to the right.'}"


# Launching the ROS2 AI Agent for the Universal Robotics (UR) robotic arm

This demo can be run using two terminals.

In the first terminal, launch the ROS2 AI Agent, and the UR robotic arm Gazebo simulation, as follows:

   - ros2 launch ros2_ai_agent start_robotic_arm_agent.launch.py

By default, all the tools are enabled.  The agent is configuration with the following launch arguments:

   - use_basic_tools:=True|False : get_ros_distro, get_domain_id
   - use_generic_tools:=True|False : list_topics, list_nodes, list_services, list_actions
   - use_robot_tools:=True|False : move_to_pose, get_current_pose, move_to_named_target
   
In the second terminal, send user prompts to the ROS2 AI Agent as strings to the /llm_prompt topic:

   - ros2 topic pub -1 /llm_prompt std_msgs/msg/String "{data: 'List the ROS topics.'}"
   - ros2 topic pub -1 /llm_prompt std_msgs/msg/String "{data: 'Move the robot to home position'}"
   - ros2 topic pub -1 /llm_prompt std_msgs/msg/String "{data: 'Get the current gripper pose'}"
   - ros2 topic pub -1 /llm_prompt std_msgs/msg/String "{data: 'Get the current position of end effector and reduce z value by 0.2'}"

# Validation Metrics

... TBD ... 

# References:

Mastering ROS2 for Robotics Applications:

   - [Packt] https://www.packtpub.com/en-us/product/mastering-ros-2-for-robotics-programming-9781836209003
   - [github] https://github.com/PacktPublishing/Mastering-ROS-2-for-Robotics-Programming/tree/main/Chapter14/ros2_basic_agent

Agentic AI with Andrew Ng

   - [DeepLearning.AI] https://learn.deeplearning.ai/courses/agentic-ai/information
   
