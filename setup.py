from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'brain_robot_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # ANN model and metadata
        (os.path.join('share', package_name, 'models'),
            glob('models/*')),

        # Processed HAPT dataset and spike/memristor results
        (os.path.join('share', package_name, 'data', 'processed'),
            glob('data/processed/*')),

        # Configuration files
        (os.path.join('share', package_name, 'config'),
            glob('config/*')),

        # Launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*')),

        # Gazebo worlds
        (os.path.join('share', package_name, 'worlds'),
            glob('worlds/*')),
    ],
    package_data={'': ['py.typed']},
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vboxuser',
    maintainer_email='vboxuser@todo.todo',
    description='Brain-inspired robotic control system using ANN and TiO2 memristor',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'brain_controller_node = brain_robot_controller.brain_controller_node:main',
        ],
    },
)
