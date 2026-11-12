import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Pobranie ścieżki do zainstalowanej paczki 'dron_sterowanie'
    pkg_share = get_package_share_directory('dron_sterowanie')

    # Automatyczne wskazanie pliku konfiguracyjnego regulator_params.yaml
    config_file = os.path.join(pkg_share, 'config', 'regulator_params.yaml')

    # Uruchomienie węzła regulatora z automatycznym załadowaniem parametrów z pliku YAML
    regulator_node = Node(
        package='dron_sterowanie',
        executable='regulator',
        name='regulator_node',
        parameters=[config_file],
        output='screen'
    )

    return LaunchDescription([
        regulator_node
    ])
