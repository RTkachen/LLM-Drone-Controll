import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Pobranie ścieżki do zainstalowanej paczki 'moj_dron' (w folderze install/moj_dron/share/moj_dron)
    pkg_share = get_package_share_directory('moj_dron')

    # Zbudowanie pełnych ścieżek do pliku świata (.sdf) oraz katalogu z modelami
    sciezka_swiata = os.path.join(pkg_share, 'worlds', 'drone_world.sdf')
    sciezka_modeli = os.path.join(pkg_share, 'models')

    # 2. Dodanie ścieżki naszych modeli do zmiennej środowiskowej GZ_SIM_RESOURCE_PATH.
    # Dzięki temu Gazebo wie, gdzie szukać modelu wpisując <uri>model://x3_uav</uri> w pliku świata.
    gz_resource_path = AppendEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=sciezka_modeli
    )

    # 3. Uruchomienie symulatora Gazebo poprzez dołączenie oficjalnego pliku launch z paczki 'ros_gz_sim'.
    # Argument '-r' (run) sprawia, że symulacja startuje od razu (bez pauzy na starcie).
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'-r {sciezka_swiata}'}.items()
    )

    # 4. Uruchomienie mostka (ros_gz_bridge) tłumaczącego komunikaty między ROS2 a Gazebo.
    # Składnia: 'nazwa_topicu@typ_wiadomosci_ROS2@typ_wiadomosci_Gazebo'
    bridges = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            # Mostkowanie komend prędkości silników (ROS2 <-> Gazebo)
            '/X3/gazebo/command/motor_speed@actuator_msgs/msg/Actuators@gz.msgs.Actuators',
            # Mostkowanie odometrii drona, czyli pozycji, orientacji i prędkości (Gazebo <-> ROS2)
            '/model/X3/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry'
        ],
        output='screen' # Wypisywanie logów mostka bezpośrednio w terminalu
    )

    # 5. Uruchomienie naszego węzła zakłócającego pracę silników (z pliku motor_disturbance.py)
    motor_disturbance = Node(
        package='moj_dron',                 # Nazwa naszej paczki
        executable='motor_disturbance',     # Nazwa punktu wejścia zdefiniowana w setup.py
        name='motor_disturbance_node',      # Nazwa węzła widoczna w systemie ROS2
        output='screen'
    )

    # Zwrócenie listy wszystkich elementów, które mają zostać kolejno uruchomione
    return LaunchDescription([
        gz_resource_path,
        gz_sim,
        bridges,
        motor_disturbance
    ])
