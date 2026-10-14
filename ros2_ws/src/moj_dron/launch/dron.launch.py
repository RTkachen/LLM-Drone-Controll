import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    sciezka_swiata = os.path.join(get_package_share_directory('moj_dron'), 'worlds', 'drone_world.sdf')

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': f'-r {sciezka_swiata}'}.items()
    )

    bridges = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            # Mostek dla silników (wyjście z PID do Gazebo)
            '/X3/gazebo/command/motor_speed@actuator_msgs/msg/Actuators@gz.msgs.Actuators',

            # Mostek dla odometrii (wejście do PID z Gazebo)
            '/model/X3/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry'
        ],
        output='screen'
    )

    return LaunchDescription([gz_sim, bridges])
