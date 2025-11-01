from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
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

    # Declare launch arguments
    llm_api_arg = DeclareLaunchArgument(
        "llm_api",
        default_value="ollama",
        description="Which API to use for LLM (openai, ollama)."
    )    
    llm_model_arg = DeclareLaunchArgument(
        "llm_model",
        default_value="gpt-oss:20b",
        description="Name of LLM format={model}:{variant} default=gpt-oss:20b"
    )
    use_basic_tools_arg = DeclareLaunchArgument(
        "use_basic_tools",
        default_value="True",
        description="Include basic tools (get_ros_distro, get_domain_id)."
    )
    use_generic_tools_arg = DeclareLaunchArgument(
        "use_generic_tools",
        default_value="True",
        description="Include generic ROS2 tools (list_topics, list_nodes, list_services, list_actions)."
    )
    use_robot_tools_arg = DeclareLaunchArgument(
        "use_robot_tools",
        default_value="True",
        description="Include robot-specific tools (move_to_pose, get_current_pose, move_to_named_target."
    )

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
                   {"llm_model":LaunchConfiguration("llm_model")},
                   {"use_basic_tools":PythonExpression(['"', LaunchConfiguration('use_basic_tools'), '" == "True"'])},
                   {"use_generic_tools":PythonExpression(['"', LaunchConfiguration('use_generic_tools'), '" == "True"'])},
                   {"use_robot_tools":PythonExpression(['"', LaunchConfiguration('use_robot_tools'), '" == "True"'])}
                ],
                output='screen',
                emulate_tty=True,
            )
        ]
    )

    return LaunchDescription([
        llm_api_arg, llm_model_arg,
        use_basic_tools_arg, use_generic_tools_arg, use_robot_tools_arg,
        ur_moveit_launch,
        delayed_agent
    ])
