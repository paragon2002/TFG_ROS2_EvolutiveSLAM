import rclpy 
from rclpy.node import Node
import sensor_msgs.msg as sensor_msgs
import nav_msgs.msg as nav_msgs
import std_msgs.msg as std_msgs
import numpy as np
import open3d as o3d
import csv
import os
import math

import warnings

from ament_index_python.packages import get_package_share_directory

from evloc.read_points import read_points
from evloc.common_classes import Color
from evloc.generate_point_cloud import generate_point_cloud
from evloc.ask_params import ask_params

from evloc.common_classes import spatial_rotation
import time

########### GLOBAL CONSTANTS ###########

home_route = os.path.expanduser("~")

SIM_LOCAL_CLOUDS_FOLDER = f"{home_route}/sim_local_clouds"

########################################

# Filter out the RuntimeWarning for invalid value encountered in divide
# For when NaN is calculated in add_noise_to_pc.
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

##############################################################

def filter_map_height(map, z_min, z_max):

    # Create a crop box to keep only the points between z_min and z_max
    crop_box = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(-float('inf'), -float('inf'), z_min),
        max_bound=(float('inf'), float('inf'), z_max)
    )

    # Apply the crop to the point cloud
    filtered_map = map.crop(crop_box)

    return filtered_map

############################################################
########################## MAIN ############################
############################################################

class PCD(Node):

    def __init__(self):
        super().__init__('pcd_node')

        self.pcd_subscriber = self.create_subscription(
            sensor_msgs.PointCloud2,
            '/velodyne_points',
            self.listener_callback,
            10  # el número de mensajes en la cola
        )

        self.odom_subscriber = self.create_subscription(
            nav_msgs.Odometry,
            '/odom',
            self.odom_callback,
            10  # el número de mensajes en la cola
        )

        self.pcd_publisher_global = self.create_publisher(sensor_msgs.PointCloud2, 'evloc_local', 10)

        self.cloud_points = None
        self.groundtruth = np.full(6, np.inf)

    def listener_callback(self, msg):
        self.cloud_points = msg

    def odom_callback(self, msg):
        self.groundtruth = msg.pose.pose

        # Extraer los valores de posición (X, Y, Z)
        x = self.groundtruth.position.x
        y = self.groundtruth.position.y
        z = self.groundtruth.position.z

        # Extraer los valores de orientación en cuaternión (qx, qy, qz, qw)
        qx = self.groundtruth.orientation.x
        qy = self.groundtruth.orientation.y
        qz = self.groundtruth.orientation.z
        qw = self.groundtruth.orientation.w

        # Convertir los cuaterniones a ángulos de Euler (A, B, C)
        # Asegúrate de que los ángulos estén en el rango adecuado (por ejemplo, -pi a pi)
        roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
        pitch = math.asin(2 * (qw * qy - qz * qx))
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

        self.groundtruth = np.array([x, y, z, roll, pitch, yaw])

    def store_groundtruth(self, groundtruth):

        filename = "sim_groundtruth_data.csv"
        filepath = os.path.join(home_route, filename)

        # Initializing CSV file with header if it doesn't exist
        if not os.path.exists(filepath):
            with open(filepath, mode='w', newline='') as archivo_csv:
                escritor_csv = csv.writer(archivo_csv)
                escritor_csv.writerow(['x','y','z','pitch','yaw','roll'])


        # Escribir los datos en el archivo CSV
        with open(filepath, mode='a', newline='') as archivo_csv:
            escritor_csv = csv.writer(archivo_csv)
            escritor_csv.writerow(groundtruth)

        print(f"Groundtruth: {groundtruth}. Guardado")
    


    def run(self):
        not_finished = self.ask_restart('Store?')
        id = 1
        while not_finished:
            rclpy.spin_once(self) # Read once from subscribed topics
            while (self.cloud_points == None or np.all(np.isinf(self.groundtruth))):
                print("Waiting for local scan...")
                rclpy.spin_once(self)
            

            # Transform map_local datatype
            points = read_points(self.cloud_points, skip_nans=True, field_names=("x", "y", "z"))
            point_list = np.array(list(points))
            map_local_unfiltered = o3d.geometry.PointCloud()
            map_local_unfiltered.points = o3d.utility.Vector3dVector(point_list)
            map_local = filter_map_height(map_local_unfiltered, 0, 1.35)

            o3d.io.write_point_cloud(f'{SIM_LOCAL_CLOUDS_FOLDER}/cloud_{id}.ply', map_local)
            self.store_groundtruth(self.groundtruth)

            self.pcd_publisher_global.publish(self.cloud_points)

            print(f"Stored sim_pc_{id} of dimensions {np.asarray(map_local.points).shape}")

            not_finished = self.ask_restart('Store?')

            self.cloud_points = None
            self.groundtruth = np.full(6, np.inf)
            id += 1

    def ask_restart(self, prompt):
        while True:
            user_input = input(f"{prompt} (y/n): ")
            if user_input == 'y':
                return True
            elif user_input == 'n':
                return False
            else:
                print("Invalid answer. Please type 'y' for yes or 'n' for no.")

def main(args=None):
    rclpy.init(args=args)
    pcd = PCD()
    pcd.run()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
