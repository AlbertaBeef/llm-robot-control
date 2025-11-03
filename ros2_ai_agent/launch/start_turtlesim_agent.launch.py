from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "llm_api",
            default_value="ollama",
            description="Which API to use for LLM (openai, ollama)."
        ),     
        DeclareLaunchArgument(
            "llm_model",
            default_value="gpt-oss:20b",
            description="Name of LLM format={model}:{variant} default=gpt-oss:20b"
        ),
        DeclareLaunchArgument(
            "use_basic_tools",
            default_value="True",
            description="Include basic tools (get_ros_distro, get_domain_id)."
        ),
        DeclareLaunchArgument(
            "use_generic_tools",
            default_value="True",
            description="Include generic ROS2 tools (list_topics, list_nodes, list_services, list_actions)."
        ),
        DeclareLaunchArgument(
            "use_robot_tools",
            default_value="True",
            description="Include robot-specific tools (move_forward, rotate, get_pose)."
        ), 
        Node(
            package='turtlesim',
            executable='turtlesim_node',
            name='turtlesim',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='ros2_ai_agent',
            executable='ros2_ai_agent_turtlesim',
            name='ros2_ai_agent_turtlesim',
            parameters=[
               {"llm_enabled":True},
               {"llm_api":LaunchConfiguration("llm_api")},
               {"llm_model":LaunchConfiguration("llm_model")},
               {"use_basic_tools":PythonExpression(['"', LaunchConfiguration('use_basic_tools'), '" == "True"'])},
               {"use_generic_tools":PythonExpression(['"', LaunchConfiguration('use_generic_tools'), '" == "True"'])},
               {"use_robot_tools":PythonExpression(['"', LaunchConfiguration('use_robot_tools'), '" == "True"'])}
            ],
            output='screen',
            emulate_tty=True,
        )
    ])
