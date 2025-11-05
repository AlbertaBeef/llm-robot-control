cmake_minimum_required(VERSION 3.8)
project(ros2_ai_interfaces)

# Build tools
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# ROSIDL for interface generation
find_package(rosidl_default_generators REQUIRED)
# If your .srv references other packages, uncomment and add them:
# find_package(std_msgs REQUIRED)
# find_package(builtin_interfaces REQUIRED)

# Install the srv sources for tooling/redistribution
install(DIRECTORY srv
  DESTINATION share/${PROJECT_NAME}
)

# Generate the service interface(s)
# Make sure you renamed the file to CamelCase: srv/SetLlmMode.srv
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SetLlmMode.srv"
  # If your .srv uses other message types, list them here:
  # DEPENDENCIES std_msgs builtin_interfaces
)
 
# Export runtime so downstream packages can use the generated types
ament_export_dependencies(rosidl_default_runtime)

ament_package()
