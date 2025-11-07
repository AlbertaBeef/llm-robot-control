from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

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
    #
    models_csv_arg = DeclareLaunchArgument(
        "models_csv",
        default_value="models.csv",
        description="Name of CSV file containing list of models to evaluate"
    )
    tasks_csv_arg = DeclareLaunchArgument(
        "tasks_csv",
        default_value="tasks.csv",
        description="Name of CSV file containing list of tasks to evaluate"
    )
    results_csv_arg = DeclareLaunchArgument(
        "results_csv",
        default_value="results.csv",
        description="Name of CSV file to store evaluation results"
    )

    # Turtlesim
    turtlesim_node = Node(
        package='turtlesim',
        executable='turtlesim_node',
        name='turtlesim',
        output='screen',
        emulate_tty=True,
    )
    
    # Define the AI agent node with a delay
    delayed_agent_node = TimerAction(
        period=1.0,  # 1 second delay
        actions=[
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
        ]
    )

    # Define the AI eval node with a delay
    delayed_eval_node= TimerAction(
        period=10.0,  # 10 second delay
        actions=[
            Node(
                package='ros2_ai_eval',
                executable='ros2_ai_eval',
                name='ros2_ai_eval',
                parameters=[
                   {"models_csv":LaunchConfiguration("models_csv")},
                   {"tasks_csv":LaunchConfiguration("tasks_csv")},
                   {"results_csv":LaunchConfiguration("results_csv")}               
                ]
            ) 
        ]
    )
            
    return LaunchDescription([
        llm_api_arg, llm_model_arg,
        use_basic_tools_arg, use_generic_tools_arg, use_robot_tools_arg,
        models_csv_arg, tasks_csv_arg, results_csv_arg,
        turtlesim_node,
        delayed_agent_node,
        delayed_eval_node
    ])    
