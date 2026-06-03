import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import open3d as o3d
import numpy as np
import os
import struct
from std_msgs.msg import Header

class OdomPublisher(Node):
    def __init__(self):
        super().__init__('odom_map_publisher')
        self.publisher_ = self.create_publisher(PointCloud2, '/mapa_odometria', 10)
        self.timer = self.create_timer(5.0, self.timer_callback)
        self.file_path = os.path.expanduser('~/ros2_ws/src/ROS2_Evolutive_Localization/tools/dataset_real/mapa_odometria_pura.ply')

    def timer_callback(self):
        if not os.path.exists(self.file_path):
            return
            
        pcd = o3d.io.read_point_cloud(self.file_path)
        points = np.asarray(pcd.points)
        
        msg = PointCloud2()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * points.shape[0]
        msg.is_dense = True
        
        buffer = bytearray(msg.row_step)
        for i, point in enumerate(points):
            struct.pack_into('<fff', buffer, i * 12, point[0], point[1], point[2])
        msg.data = memoryview(buffer).tobytes()
        
        self.publisher_.publish(msg)
        print("Mapa de Odometría publicado en RViz (/mapa_odometria)")

def main(args=None):
    rclpy.init(args=args)
    node = OdomPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
