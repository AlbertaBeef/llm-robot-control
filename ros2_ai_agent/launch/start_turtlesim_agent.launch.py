from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
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
               {"llm_api":LaunchConfiguration("llm_api")},
               {"llm_model":LaunchConfiguration("llm_model")}
            ],
            output='screen',
            emulate_tty=True,
        )
    ])
