import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'moj_dron'

# Rekurencyjne przeszukiwanie folderu 'models' (funkcja os.walk).
# Tworzy listę krotek: (ścieżka_docelowa_w_install, [lista_plików_źródłowych]).
# Jest to niezbędne, aby podczas 'colcon build' skopiować model drona (model.sdf, model.config) do folderu install/.
model_files = [
    (os.path.join('share', package_name, root), [os.path.join(root, f) for f in files])
    for root, _, files in os.walk('models') if files
]

setup(
    name=package_name,
    version='0.0.0',
    # Automatyczne wykrywanie modułów Pythona (folderów z plikiem __init__.py), z pominięciem folderu 'test'
    packages=find_packages(exclude=['test']),

    # Lista dodatkowych plików (niebędących kodem Pythona), które muszą trafić do folderu install/
    data_files=[
        # Rejestracja paczki w indeksie ament (wymagane przez ROS2, aby widział paczkę)
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # Kopiowanie pliku manifestu package.xml
        ('share/' + package_name, ['package.xml']),
        # Kopiowanie wszystkich plików uruchomieniowych (*.launch.py) z folderu launch/
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # Kopiowanie wszystkich plików świata (*.sdf) z folderu worlds/
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
    ] + model_files, # Doklejenie wcześniej wygenerowanej listy plików modelu drona

    package_data={'': ['py.typed']},
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='RTkachen',
    maintainer_email='rtkachenko@tuta.io',
    description='Środowisko symulacyjne drona X3 z zakłóceniami',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    # Definicja wykonywalnych skryptów (węzłów ROS2), które można uruchamiać przez 'ros2 run moj_dron <nazwa>'
    entry_points={
        'console_scripts': [
            # 'nazwa_wykonywalna = nazwa_folderu.nazwa_pliku_bez_py:funkcja_startowa'
            'motor_disturbance = moj_dron.motor_disturbance:main',
        ],
    },
)
