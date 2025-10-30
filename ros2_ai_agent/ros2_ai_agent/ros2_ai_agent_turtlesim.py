#!/usr/bin/env python3

"""
This script defines a ROS2 node that integrates with OpenAI's language model to control a turtle in the turtlesim simulation.
The node subscribes to a topic and processes incoming messages using an AI agent with predefined tools.

Acknowledgements:
    This code is based on the following source code:
    - https://github.com/PacktPublishing/Mastering-ROS-2-for-Robotics-Programming/tree/main/Chapter14/ros2_basic_agent
    
Classes:
    ROS2AIAgent(Node): A ROS2 node that subscribes to a topic and uses an AI agent to control the turtle.

Key Methods:
    - prompt_callback(msg: String): Handles incoming messages and processes them using the AI agent.
    - get_ros_distro() -> str: Retrieves the current ROS distribution name.
    - get_domain_id() -> str: Retrieves the current ROS domain ID.
Methods:
    prompt_callback(msg: String): Callback function to process incoming messages.
    list_topics() -> str: Lists all available ROS 2 topics.
    list_nodes() -> str: Lists all running ROS 2 nodes.
    list_services() -> str: Lists all available ROS 2 services.
    list_actions() -> str: Lists all available ROS 2 actions.
Functions:
    move_forward(distance: float) -> str: Moves the turtle forward by the specified distance.
    rotate(angle: float) -> str: Rotates the turtle by the specified angle in degrees.
    get_pose() -> str: Gets the current position and orientation of the turtle.
    pose_callback(msg: Pose): Callback function to update the turtle's pose.
    prompt_callback(msg: String): Callback function to process incoming messages.
    main(args=None): Initializes and spins the ROS2 node.

Dependencies:
    - os
    - math
    - geometry_msgs.msg (Twist)
    - turtlesim.msg (Pose)
    - langchain.agents (AgentExecutor, create_openai_tools_agent)
    - langchain_openai (ChatOpenAI)
    - langchain_ollama (ChatOllama)
    - langchain.tools (BaseTool, StructuredTool, tool)
    - langchain_core.prompts (ChatPromptTemplate, MessagesPlaceholder)
    - dotenv (load_dotenv)
    - std_msgs.msg (String)
    - rclpy (rclpy, Node)
"""

import os
import math
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from std_msgs.msg import String
import rclpy
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory, get_package_prefix
from pathlib import Path

from transforms3d.euler import quat2euler  # Replace tf_transformations import
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

import subprocess
from typing import List


class ROS2AIAgent(Node):
    def __init__(self):
        super().__init__('ros2_ai_agent')
        self.get_logger().info('ROS2 AI Agent has been started')

        # LLM parameters        
        self.declare_parameter("llm_api", "ollama")
        self.llm_api = self.get_parameter("llm_api").value
        self.get_logger().info('llm_api : "%s"' % self.llm_api)
        #
        self.declare_parameter("llm_model", "gpt-oss:20b")
        self.llm_model = self.get_parameter("llm_model").value
        self.get_logger().info('llm_model : "%s"' % self.llm_model)
        
        # Initialize turtle pose
        self.turtle_pose = Pose()
        
        # Create publisher for turtle commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        
        # Create subscriber for turtle pose
        self.pose_sub = self.create_subscription(
            Pose,
            '/turtle1/pose',
            self.pose_callback,
            10
        )

        @tool
        def get_ros_distro() -> str:
            """Get the current ROS distribution name."""
            try:
                ros_distro = os.environ.get('ROS_DISTRO')
                if (ros_distro):
                    return f"Current ROS distribution: {ros_distro}"
                else:
                    return "ROS distribution environment variable (ROS_DISTRO) not set"
            except Exception as e:
                return f"Error getting ROS distribution: {str(e)}"

        @tool
        def get_domain_id() -> str:
            """Get the current ROS domain ID."""
            try:
                domain_id = os.environ.get('ROS_DOMAIN_ID', '0')  # Default is 0 if not set
                return f"Current ROS domain ID: {domain_id}"
            except Exception as e:
                return f"Error getting ROS domain ID: {str(e)}"

        @tool
        def list_topics() -> str:
            """List all available ROS 2 topics."""
            try:
                result = subprocess.run(['ros2', 'topic', 'list'], 
                                 capture_output=True, text=True, check=True)
                topics = result.stdout.strip().split('\n')
                return f"Available ROS 2 topics:\n{result.stdout}"
            except subprocess.CalledProcessError as e:
                return f"Error listing topics: {str(e)}"

        @tool
        def list_nodes() -> str:
            """List all running ROS 2 nodes."""
            try:
                result = subprocess.run(['ros2', 'node', 'list'], 
                                 capture_output=True, text=True, check=True)
                return f"Running ROS 2 nodes:\n{result.stdout}"
            except subprocess.CalledProcessError as e:
                return f"Error listing nodes: {str(e)}"

        @tool
        def list_services() -> str:
            """List all available ROS 2 services."""
            try:
                result = subprocess.run(['ros2', 'service', 'list'], 
                                 capture_output=True, text=True, check=True)
                return f"Available ROS 2 services:\n{result.stdout}"
            except subprocess.CalledProcessError as e:
                return f"Error listing services: {str(e)}"

        @tool
        def list_actions() -> str:
            """List all available ROS 2 actions."""
            try:
                result = subprocess.run(['ros2', 'action', 'list'], 
                                 capture_output=True, text=True, check=True)
                return f"Available ROS 2 actions:\n{result.stdout}"
            except subprocess.CalledProcessError as e:
                return f"Error listing actions: {str(e)}"

        # setup the tools as class methods
        @tool
        def move_forward(distance: float) -> str:
            """Move turtle forward by specified distance."""
            msg = Twist()
            duration = float(distance)

            msg.linear.x = duration * 0.1
            
            # Publish for the calculated duration
            self.cmd_vel_pub.publish(msg)
            #self.create_timer(duration, lambda: self.cmd_vel_pub.publish(Twist()))
            #self.create_timer(duration, lambda: self.cmd_vel_pub.publish(msg))
            return f"Moved forward {distance} units"

        @tool
        def rotate(angle: float) -> str:
            """Rotate turtle by specified angle in degrees (positive for counterclockwise)."""
            msg = Twist()
            msg.angular.z = math.radians(float(angle))
            duration = 1.0  # Time to complete rotation
            
            self.cmd_vel_pub.publish(msg)
            self.create_timer(duration, lambda: self.cmd_vel_pub.publish(Twist()))
            return f"Rotated {angle} degrees"

        @tool
        def get_pose() -> str:
            """Get current pose of the turtle."""
            return f"x: {self.turtle_pose.x:.2f}, y: {self.turtle_pose.y:.2f}, theta: {math.degrees(self.turtle_pose.theta):.2f} degrees"

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a turtle control assistant for ROS 2 turtlesim.
            You can check ROS 2 system status using these commands:
            - get_ros_distro(): Get the current ROS distribution name
            - get_domain_id(): Get the current ROS_DOMAIN_ID
            You can check ROS 2 system status using these commands:
            - list_topics(): List all available ROS 2 topics
            - list_nodes(): List all running ROS 2 nodes
            - list_services(): List all available ROS 2 services
            - list_actions(): List all available ROS 2 actions
              You can control the turtle using these commands:
            - move_forward(distance): Move turtle forward by specified distance
            - rotate(angle): Rotate turtle by specified angle in degrees
            - get_pose(): Get current position and orientation of turtle
            
            Return only the necessary actions and their results. e.g
            Human: What ROS distribution am I using?
            AI: Current ROS distribution: humble
            Human: What is my ROS domain ID?
            AI: Current ROS domain ID: 0
            Human: Show me all running nodes
            AI: Here are the running ROS 2 nodes: [node list]
            Human: Move the turtle forward 2 units
            AI: Moving forward 2 units
            """),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        share_dir = get_package_share_directory('ros2_ai_agent')
        config_dir = share_dir + '/config' + '/openai.env'
        load_dotenv(Path(config_dir))

        # setup the toolkit with the class methods
        self.toolkit = [
            get_ros_distro, get_domain_id,
            list_topics, list_nodes, list_services, list_actions,
            move_forward, rotate, get_pose]

        # Choose the LLM that will drive the agent
        
        # 
        # GPT-5
        #
        # [ros2_ai_agent_turtlesim-2] [ERROR] [1759210085.895728632] [ros2_ai_agent_turtlesim]: Error processing prompt: Error code: 400 - {'error': {'message': 'Your organization must be verified to stream this model. Please go to: https://platform.openai.com/settings/organization/general and click on Verify Organization. If you just verified, it can take up to 15 minutes for access to propagate.', 'type': 'invalid_request_error', 'param': 'stream', 'code': 'unsupported_value'}}
        #self.llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
        #self.llm = ChatOpenAI(model="gpt-5", temperature=0)

        #
        # GPT-4.1
        #
        #self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

        #
        # GPT-4o
        #
        #self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        #self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

        #
        # GPT-OSS
        #
        #self.llm = ChatOllama( model="gpt-oss", temperature=0, num_ctx=131072 )
        
        if self.llm_api == "openai":
            self.llm = ChatOpenAI(model=self.llm_model, temperature=0)
        elif self.llm_api == "ollama":
            self.llm = ChatOllama( model=self.llm_model, temperature=0 )
        else:
            self.get_logger().error(f'Invalid llm_api: {self.llm_api}')

        # Construct the OpenAI Tools agent
        self.agent = create_openai_tools_agent(self.llm, self.toolkit, self.prompt)

        # Create an agent executor by passing in the agent and tools
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.toolkit, verbose=True)

        # Create the subscriber for prompts
        self.subscription = self.create_subscription(
            String,
            'prompt',
            self.prompt_callback,
            10
        )

    def pose_callback(self, msg):
        """Callback to update turtle's pose"""
        self.turtle_pose = msg

    def prompt_callback(self, msg):
        try:
            result = self.agent_executor.invoke({"input": msg.data})
            self.get_logger().info(f"Result: {result['output']}")
        except Exception as e:
            self.get_logger().error(f'Error processing prompt: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = ROS2AIAgent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
