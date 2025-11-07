from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'ros2_ai_eval'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*'))  # Add this line

    ],
    install_requires=['setuptools','langchain','langchain-openai','langchain-ollama','python-dotenv'],
    zip_safe=True,
    maintainer='AlbertaBeef',
    maintainer_email='grouby177@gmail.com',
    description='ROS 2 AI evaluation package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ros2_ai_eval = ros2_ai_eval.ros2_ai_eval:main',            
        ],
    },
)
