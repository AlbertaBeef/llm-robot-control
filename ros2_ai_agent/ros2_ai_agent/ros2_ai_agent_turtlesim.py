#!/usr/bin/env python3

"""
This script defines a ROS2 node that integrates with OpenAI's language model to control a turtle in the turtlesim simulation.
The node subscribes to a topic and processes incoming messages using an AI agent with predefined tools.

Acknowledgements:
    This code is based on the following source code:
    - https://github.com/PacktPublishing/Mastering-ROS-2-for-Robotics-Programming/tree/main/Chapter14/ros2_basic_agent
    
Classes:
    ROS2AIAgent(Node): A ROS2 node that subscribes to a topic and uses an AI agent to control the turtle.

Functions:
    Basic (ROS2 specific) Tools:
    - get_ros_distro() -> str: Retrieves the current ROS distribution name.
    - get_domain_id() -> str: Retrieves the current ROS domain ID.
    General (ROS2 specific) Tools:
    - list_topics() -> str: Lists all available ROS 2 topics.
    - list_nodes() -> str: Lists all running ROS 2 nodes.
    - list_services() -> str: Lists all available ROS 2 services.
    - list_actions() -> str: Lists all available ROS 2 actions.
    Robot specific Tools:
    - move_forward(distance: float) -> str: Moves the turtle forward by the specified distance.
    - rotate(angle: float) -> str: Rotates the turtle by the specified angle in degrees.
    - get_pose() -> str: Gets the current position and orientation of the turtle.
    Other functions:
    - pose_callback(msg: Pose): Callback function to update the turtle's pose.
    - prompt_callback(msg: String): Callback function to process incoming messages.
    - main(args=None): Initializes and spins the ROS2 node.

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
from langchain.agents import AgentExecutor, create_openai_tools_agent, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

import subprocess
from typing import List

from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

# Import the generated service type
try:
    from ros2_ai_agent.srv import SetLlmMode
except Exception:
    SetLlmMode = None

class ROS2AIAgent(Node):
    # --------------------------
    # ROS2AIAgent constructor
    # --------------------------
    def __init__(self):
        super().__init__('ros2_ai_agent')
        self.get_logger().info('ROS2 AI Agent has been started')

        # LLM parameters
        self.declare_parameter('llm_enabled', True)
        self.llm_enabled = self.get_parameter("llm_enabled").value
        self.get_logger().info('llm_enabled : "%s"' % self.llm_enabled)
        #
        self.declare_parameter("llm_api", "ollama")
        self.llm_api = self.get_parameter("llm_api").value
        self.get_logger().info('llm_api : "%s"' % self.llm_api)
        #
        self.declare_parameter("llm_model", "gpt-oss:20b")
        self.llm_model = self.get_parameter("llm_model").value
        self.get_logger().info('llm_model : "%s"' % self.llm_model)

        # Tools parameters
        self.declare_parameter("use_basic_tools", True)
        self.use_basic_tools = self.get_parameter("use_basic_tools").value
        self.get_logger().info('use_basic_tools : "%s"' % self.use_basic_tools)
        #        
        self.declare_parameter("use_generic_tools", True)
        self.use_generic_tools = self.get_parameter("use_generic_tools").value
        self.get_logger().info('use_generic_tools : "%s"' % self.use_generic_tools)
        #
        self.declare_parameter("use_robot_tools", True)
        self.use_robot_tools = self.get_parameter("use_robot_tools").value
        self.get_logger().info('use_robot_tools : "%s"' % self.use_robot_tools)
        
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

        # Create System Prompt and LLM+Agent based on parameters
        self.system_prompt_creator()
        self.toolkit_creator()
        self.llm_agent_creator()        

        # Create the subscriber for prompts
        self.subscription = self.create_subscription(
            String,
            'llm_prompt',
            self.llm_prompt_callback,
            10
        )

        # Create the publisher for Tool usage confirmation
        self.llm_tool_calls_pub = self.create_publisher(String, '/llm_tool_calls', 10)

        # Create the publisher for LLM output
        self.llm_output_pub = self.create_publisher(String, '/llm_output', 10)

        # Validate parameter changes done via `ros2 param set`
        self.add_on_set_parameters_callback(self.on_set_parameters_callback)

        # LLM toggle service
        if SetLlmMode is not None:
            self.set_llm_srv = self.create_service(
                SetLlmMode,
                'set_llm_mode',
                self.set_llm_mode_callback
            )
            self.get_logger().info('LLM state service /set_llm_mode ready (type: ros2_ai_agent/srv/SetLlmMode)')
        else:
            self.get_logger().warn(
                'Service /SetLlmMode not available yet. Build the package (colcon build) to enable the service.'
            )

    # --------------------------
    # System Prompt creator
    # --------------------------
    def system_prompt_creator(self):
        if self.use_basic_tools == True:
            basic_tools_prompt1 = """
            You can check ROS 2 system status using these commands:
            - get_ros_distro(): Get the current ROS distribution name
            - get_domain_id(): Get the current ROS_DOMAIN_ID
            """    
            basic_tools_prompt2 = """
            Human: What ROS distribution am I using?
            AI: Current ROS distribution: humble
            Human: What is my ROS domain ID?
            AI: Current ROS domain ID: 0
            Human: Show me all running nodes
            """
        else:
            basic_tools_prompt1 = ""
            basic_tools_prompt2 = ""

        if self.use_generic_tools == True:
            generic_tools_prompt1 = """
            You can check ROS 2 system status using these commands:
            - list_topics(): List all available ROS 2 topics
            - list_nodes(): List all running ROS 2 nodes
            - list_services(): List all available ROS 2 services
            - list_actions(): List all available ROS 2 actions
            """    
            generic_tools_prompt2 = """
            Human: Show me all running nodes
            AI: Here are the running ROS 2 nodes: [node list]
            """
        else:
            generic_tools_prompt1 = ""
            generic_tools_prompt2 = ""

        if self.use_robot_tools == True:
            robot_tools_prompt1 = """
            You can control the turtle using these commands:
            - move_forward(distance): Move turtle forward by specified distance
            - rotate(angle): Rotate turtle by specified angle in degrees
            - get_pose(): Get current position and orientation of turtle
            """    
            robot_tools_prompt2 = """
            Human: Move the turtle forward 2 units
            AI: Moving forward 2 units
            """
        else:
            robot_tools_prompt1 = ""
            robot_tools_prompt2 = ""

        if self.use_robot_tools == False:
            self.system_prompt = """
            You are a ROS 2 system information assistant.
            """ + basic_tools_prompt1 + generic_tools_prompt1 + robot_tools_prompt1 + """
            
            Return only the necessary actions and their results. e.g
            """ + basic_tools_prompt2 + generic_tools_prompt2 + robot_tools_prompt2 + """
            """ 
        else:                 
            self.system_prompt = """
            You are a turtle control assistant for ROS 2 turtlesim.
            """ + basic_tools_prompt1 + generic_tools_prompt1 + robot_tools_prompt1 + """
            
            Return only the necessary actions and their results. e.g
            """ + basic_tools_prompt2 + generic_tools_prompt2 + robot_tools_prompt2 + """
            """ 
        self.get_logger().info('system_prompt : "%s"' % self.system_prompt)

    # --------------------------
    # Robot Tools
    # --------------------------
    def pose_callback(self, msg):
        """Callback to update turtle's pose"""
        self.turtle_pose = msg

    # --------------------------
    # Toolkit creator
    # --------------------------
    def toolkit_creator(self):

        @tool
        def get_ros_distro() -> str:
            """Get the current ROS distribution name."""

            msg = String()
            msg.data = f"get_ros_distro()"
            self.llm_tool_calls_pub.publish(msg)

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

            msg = String()
            msg.data = f"get_domain_id()"
            self.llm_tool_calls_pub.publish(msg)

            try:
                domain_id = os.environ.get('ROS_DOMAIN_ID', '0')  # Default is 0 if not set
                return f"Current ROS domain ID: {domain_id}"
            except Exception as e:
                return f"Error getting ROS domain ID: {str(e)}"

        @tool
        def list_topics() -> str:
            """List all available ROS 2 topics."""

            msg = String()
            msg.data = f"list_topics()"
            self.llm_tool_calls_pub.publish(msg)

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

            msg = String()
            msg.data = f"list_nodes()"
            self.llm_tool_calls_pub.publish(msg)

            try:
                result = subprocess.run(['ros2', 'node', 'list'], 
                                 capture_output=True, text=True, check=True)
                return f"Running ROS 2 nodes:\n{result.stdout}"
            except subprocess.CalledProcessError as e:
                return f"Error listing nodes: {str(e)}"

        @tool
        def list_services() -> str:
            """List all available ROS 2 services."""

            msg = String()
            msg.data = f"list_services()"
            self.llm_tool_calls_pub.publish(msg)

            try:
                result = subprocess.run(['ros2', 'service', 'list'], 
                                 capture_output=True, text=True, check=True)
                return f"Available ROS 2 services:\n{result.stdout}"
            except subprocess.CalledProcessError as e:
                return f"Error listing services: {str(e)}"

        @tool
        def list_actions() -> str:
            """List all available ROS 2 actions."""

            msg = String()
            msg.data = f"list_actions()"
            self.llm_tool_calls_pub.publish(msg)

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

            msg = String()
            msg.data = f"move_forward({distance})"
            self.llm_tool_calls_pub.publish(msg)

            msg = Twist()
            msg.linear.x = distance
            
            # Publish for the calculated duration
            self.cmd_vel_pub.publish(msg)
            return f"Moved forward {distance} units"

        @tool
        def rotate(angle: float) -> str:
            """Rotate turtle by specified angle in degrees (positive for counterclockwise)."""

            msg = String()
            msg.data = f"rotate({angle})"
            self.llm_tool_calls_pub.publish(msg)

            msg = Twist()
            msg.angular.z = math.radians(float(angle))
            duration = 1.0  # Time to complete rotation
            
            self.cmd_vel_pub.publish(msg)
            self.create_timer(duration, lambda: self.cmd_vel_pub.publish(Twist()))
            return f"Rotated {angle} degrees"

        @tool
        def get_pose() -> str:
            """Get current pose of the turtle."""

            msg = String()
            msg.data = f"get_pose()"
            self.llm_tool_calls_pub.publish(msg)

            return f"x: {self.turtle_pose.x:.2f}, y: {self.turtle_pose.y:.2f}, theta: {math.degrees(self.turtle_pose.theta):.2f} degrees"

        # Setup toolkit
        self.toolkit = [
            get_ros_distro, get_domain_id,
            list_topics, list_nodes, list_services, list_actions,
            move_forward, rotate, get_pose
        ]

    # --------------------------
    # LLM + Agent creator
    # --------------------------
    def llm_agent_creator(self):
    
    
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        # Load OpenAI configuration
        share_dir = get_package_share_directory('ros2_ai_agent')
        config_dir = share_dir + '/config' + '/openai.env'
        load_dotenv(Path(config_dir))

        # Choose the LLM that will drive the agent
        if self.llm_api == "openai":
            self.llm = ChatOpenAI(model=self.llm_model, temperature=0)
        elif self.llm_api == "ollama":
            self.llm = ChatOllama( model=self.llm_model, temperature=0 )
        else:
            self.get_logger().error(f'Invalid llm_api: {self.llm_api}')

        # Construct the AI Tools agent
        if self.llm_api == "openai":
            self.agent = create_openai_tools_agent(self.llm, self.toolkit, self.prompt)
        elif self.llm_api == "ollama":
            self.agent = create_tool_calling_agent(self.llm, self.toolkit, self.prompt)

        # Create an agent executor by passing in the agent and tools
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.toolkit, verbose=True)

    # --------------------------
    # Parameter change validator
    # --------------------------
    def on_set_parameters_callback(self, params: List[Parameter]) -> SetParametersResult:
        result = SetParametersResult()
        result.successful = True
        result.reason = ''

        # Preview new values to cross-validate
        next_enabled = self.llm_enabled
        next_api = self.llm_api
        next_model = self.llm_model
        for p in params:
            if p.name == 'llm_enabled' and p.type_ == Parameter.Type.BOOL:
                next_enabled = p.value
            elif p.name == 'llm_api' and p.type_ == Parameter.Type.STRING:
                next_api = p.value
            elif p.name == 'llm_model' and p.type_ == Parameter.Type.STRING:
                next_model = p.value

        # If enabling, require non-empty api and model
        if next_enabled and (not next_api or not next_model):
            result.successful = False
            result.reason = 'Enabling LLM requires non-empty llm_api and llm_model.'
            return result

        # Passed validation — update mirrors for those included in this set
        for p in params:
            if p.name == 'llm_enabled' and p.type_ == Parameter.Type.BOOL:
                self.llm_enabled = p.value
            elif p.name == 'llm_api' and p.type_ == Parameter.Type.STRING:
                self.llm_api = p.value
            elif p.name == 'llm_model' and p.type_ == Parameter.Type.STRING:
                self.llm_model = p.value

        return result

    # --------------------------
    # Service handler
    # --------------------------
    def set_llm_mode_callback(self, request, response):
        """
        SetLlmMode.srv:
          bool enable
          string llm_api
          string llm_model
          ---
          bool success
          string message
        """
        enable = bool(request.enable)
        api = request.llm_api.strip() if hasattr(request, 'llm_api') else ''
        model = request.llm_model.strip() if hasattr(request, 'llm_model') else ''

        if enable and (not api or not model):
            response.success = False
            response.message = 'When enabling, llm_api and llm_model must be provided.'
            return response

        # Apply via atomic parameter update so events/validators run consistently
        changes: List[Parameter] = [Parameter('llm_enabled', Parameter.Type.BOOL, enable)]
        if enable:
            changes.extend([
                Parameter('llm_api', Parameter.Type.STRING, api),
                Parameter('llm_model', Parameter.Type.STRING, model),
            ])

        result = self.set_parameters_atomically(changes)
        if not result.successful:
            response.success = False
            response.message = f'Failed to update parameters: {result.reason}'
            return response

        # Mirrors update in the callback; but ensure they reflect latest values
        self.llm_enabled = enable
        if enable:
            self.llm_api = api
            self.llm_model = model

            # Create System Prompt and LLM+Agent based on parameters
            self.system_prompt_creator()
            self.toolkit_creator()
            self.llm_agent_creator()        
        else:
            # There does not seem to be a way to close/cleanup LLM + Agent
            self.llm_api = None
            self.llm_model = None

        state = 'enabled' if self.llm_enabled else 'disabled'
        extra = f' (api=\"{self.llm_api}\", model=\"{self.llm_model}\")' if self.llm_enabled else ''
        self.get_logger().info(f'LLM {state}{extra}')
        response.success = True
        response.message = f'LLM {state}'
        return response

    # --------------------------
    # LLM prompt Callback
    # --------------------------
    def llm_prompt_callback(self, msg):
        try:
            result = self.agent_executor.invoke({"input": msg.data})
            self.get_logger().info(f"Output: {result['output']}")

            msg = String()
            msg.data = f"Output: ({result['output']})"
            self.llm_output_pub.publish(msg)

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
        rclpy.shutdown()

if __name__ == '__main__':
    main()
