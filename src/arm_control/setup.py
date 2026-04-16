from setuptools import find_packages, setup
import os 
from glob import glob
package_name = 'arm_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='roman',
    maintainer_email='roman@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
   entry_points={
    'console_scripts': [
        'arm_control_node = arm_control.control_node:main',
        'ik_node = arm_control.ik_node:main',
        'system_check_node = arm_control.system_check_node:main',
        'js_test_pub=arm_control.js_test_pub:main',
    ],
    },
)
