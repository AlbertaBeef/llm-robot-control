from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Include the UR MoveIt2 simulation launch file
    ur_launch_file_dir = os.path.join(get_package_share_directory('ur_simulation_gz'), 'launch')
    ur_moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([ur_launch_file_dir, '/ur_sim_moveit.launch.py'])
    )

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

    # Define the AI agent node with a delay
    delayed_agent = TimerAction(
        period=10.0,  # 10 second delay
        actions=[
            Node(
                package='ros2_ai_agent',
                executable='ros2_ai_agent_robotic_arm',
                name='ros2_ai_agent_robotic_arm',
                parameters=[
                   {"llm_api":LaunchConfiguration("llm_api")},
                   {"llm_model":LaunchConfiguration("llm_model")}
                ],
                output='screen',
                emulate_tty=True,
            )
        ]
    )

    return LaunchDescription([
        ur_moveit_launch,
        delayed_agent
    ])
