from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='brain_robot_controller',
            executable='brain_controller_node',
            name='brain_controller',
            output='screen',
            remappings=[
                ('/cmd_vel', '/model/vehicle_blue/cmd_vel')
            ],
        ),
    ])
