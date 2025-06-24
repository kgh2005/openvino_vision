import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    model_path = os.path.join(
        get_package_share_directory('openvino_vision'),
        'model',
        'best.xml'
    )

    return LaunchDescription([
        Node(
            package='openvino_vision',
            executable='openvino_vision_node',
            name='openvino_vision',
            output='screen',
            parameters=[
                {'model_xml': model_path}
            ]
        )
    ])
