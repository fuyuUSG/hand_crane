from setuptools import find_packages, setup

package_name = 'hand_crane'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test', 'windows']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='student',
    maintainer_email='zettaigoukakuuts1@gmail.com',
    description='Hand-gesture teleoperation of CRANE+ V2 with MediaPipe',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'hand_teleop_node = hand_crane.hand_teleop_node:main',
        ],
    },
)
