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
import time 

import warnings

from ament_index_python.packages import get_package_share_directory

from evloc.common_classes import Color

########### GLOBAL CONSTANTS ###########

PACKAGE_PATH = os.path.join(get_package_share_directory('evloc'), 'resources')

GROUNDTRUTH_FILE_PATH = f"{PACKAGE_PATH}/sim_groundtruth_data.csv"
LOCAL_CLOUDS_FOLDER = f"{PACKAGE_PATH}/sim_local_clouds"

GLOBAL_OFFSET = [0.2,0.1,0,0,0,0.1]

########################################

# Filter out the RuntimeWarning for invalid value encountered in divide
# For when NaN is calculated in add_noise_to_pc.
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

def spatial_rotation(point, p):
    """
    spatialTransformation
    The point gets transform by multiplying by the coordinate frame of the
    first scan according to the parameters
    Inputs:
      point: Horizontal vector nx3
      p:  Vertical vector 6x1 (rad)
    Outputs:
      transformed: Horizontal vector nx3 with the point transformed in the
      space
    """

    cAlpha = np.cos(p[3])
    sAlpha = np.sin(p[3])
    cBeta = np.cos(p[4])
    sBeta = np.sin(p[4])
    cGamma = np.cos(p[5])
    sGamma = np.sin(p[5])

    rotation_matrix = np.array([
        [cBeta * cGamma, -cBeta * sGamma, sBeta],
        [cAlpha * sGamma + cGamma * sAlpha * sBeta, cAlpha * cGamma - sAlpha * sBeta * sGamma, -cBeta * sAlpha],
        [sAlpha * sGamma - cAlpha * cGamma * sBeta, cGamma * sAlpha + cAlpha * sBeta * sGamma, cAlpha * cBeta]
    ])

    transformed = np.dot(point, rotation_matrix.T) + np.array([p[0], p[1], p[2]])

    return transformed

def filter_map_height(map, z_min, z_max):

    # Create a crop box to keep only the points between z_min and z_max
    crop_box = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(-float('inf'), -float('inf'), z_min),
        max_bound=(float('inf'), float('inf'), z_max)
    )

    # Apply the crop to the point cloud
    filtered_map = map.crop(crop_box)

    return filtered_map


def get_groundtruth_data(GROUNDTRUTH_FILE_PATH, id_cloud):
    """
    Reads the row "id_cloud" from the GROUNDTRUTH_FILE_PATH and returns it.
    """
    try:
        with open(GROUNDTRUTH_FILE_PATH, 'r') as file:
            csv_reader = csv.reader(file)
            
            for _ in range(int(id_cloud)):
                next(csv_reader)
            
            groundtruth_str = next(csv_reader)
            groundtruth = np.array(groundtruth_str, dtype=float)

            return groundtruth
        
    except FileNotFoundError:
        print("El archivo CSV no fue encontrado.")
    except StopIteration:
        print("La fila especificada excede el número de filas en el archivo CSV.")

class PCD(Node):

    def __init__(self):
        super().__init__('pcd_node')

        self.declare_parameter('auto', False)

        auto_color = Color.RED

        self.auto_mode = self.get_parameter('auto').value

        if self.auto_mode:
            auto_color = Color.GREEN

        print(auto_color + f"\nAuto Mode: {self.auto_mode}" + Color.END)

        self.pcd_publisher_local = self.create_publisher(sensor_msgs.PointCloud2, 'evloc_local', 10)
        self.pcd_publisher_global = self.create_publisher(sensor_msgs.PointCloud2, 'evloc_global', 10)

        self.auto_mode = False # CAMBIAR A MANO

    def run(self):
        id_cloud = 0

        map_global_unfiltered = o3d.io.read_point_cloud(f"{PACKAGE_PATH}/map_global_sim.pcd")
        map_global = filter_map_height(map_global_unfiltered, 0, 1.35)

        points2 = np.asarray(map_global.points)
        points2 = spatial_rotation(map_global.points, GLOBAL_OFFSET)
        pcd_global = self.point_cloud(points2, 'base_footprint')
        self.pcd_publisher_global.publish(pcd_global)
        print(f"Global PointCloud with dimensions {points2.shape} has been published.")

        while True:
            
            if not self.auto_mode:
                id_cloud = None
                selected = False
                # Local Cloud
                while not selected:
                    num_clouds = len(os.listdir(LOCAL_CLOUDS_FOLDER))

                    print(Color.BOLD + f'\nAvailable scans [1-{num_clouds}]' + Color.END)
                    id_cloud = input(Color.BOLD + "Select cloud as real scan: " + Color.END)
                    try:
                        if int(id_cloud) > num_clouds or int(id_cloud) < 1:
                            print(f'Error. Selected cloud ({id_cloud}) does not exist.') 
                        else:
                            id_cloud = int(id_cloud)
                            selected = True

                    except ValueError as e:
                        print(f'Error: Invalid Number. {e}')
            else:
                id_cloud = id_cloud + 1
                if id_cloud > 44:
                    id_cloud = 1
                    while True:
                        user_input = input("Restart? (y/n): ")
                        if user_input == 'y':
                            pass
                        elif user_input == 'n':
                            exit(1)
                        else:
                            print("Invalid answer. Please type 'y' for yes or 'n' for no.")

                time.sleep(0.5)

            real_groundtruth = get_groundtruth_data(GROUNDTRUTH_FILE_PATH, id_cloud)              

            print(Color.BOLD + "\n------------------------------------" + Color.END)

            print(f"{Color.BOLD} Cloud: {id_cloud} {Color.END}")

            real_scan_ori = o3d.io.read_point_cloud(f"{PACKAGE_PATH}/sim_local_clouds/cloud_{id_cloud}.ply")
            map_local = real_scan_ori

            points = spatial_rotation(map_local.points, real_groundtruth)
            self.publish_point_clouds(points, 'base_footprint')

            print(Color.BOLD + "\n------------------------------------" + Color.END)


    def publish_point_clouds(self, points, parent_frame):

        pcd = self.point_cloud(points, parent_frame)
        self.pcd_publisher_local.publish(pcd)
        print(f"Local PointCloud with dimensions {points.shape} has been published.")


    def ask_restart(self):
        while True:
            user_input = input("Restart? (y/n): ")
            if user_input == 'y':
                return True
            elif user_input == 'n':
                return False
            else:
                print("Invalid answer. Please type 'y' for yes or 'n' for no.")

    def point_cloud(self, points, parent_frame):
        """ Creates a point cloud message.
        Args:
            points: Nx3 array of xyz positions.
            parent_frame: frame in which the point cloud is defined
        Returns:
            sensor_msgs/PointCloud2 message

        Code source:
            https://gist.github.com/pgorczak/5c717baa44479fa064eb8d33ea4587e0

        References:
            http://docs.ros.org/melodic/api/sensor_msgs/html/msg/PointCloud2.html
            http://docs.ros.org/melodic/api/sensor_msgs/html/msg/PointField.html
            http://docs.ros.org/melodic/api/std_msgs/html/msg/Header.html

        """
        # In a PointCloud2 message, the point cloud is stored as an byte 
        # array. In order to unpack it, we also include some parameters 
        # which desribes the size of each individual point.
        ros_dtype = sensor_msgs.PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize # A 32-bit float takes 4 bytes.

        data = points.astype(dtype).tobytes() 

        # The fields specify what the bytes represents. The first 4 bytes 
        # represents the x-coordinate, the next 4 the y-coordinate, etc.
        fields = [sensor_msgs.PointField(
            name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
            for i, n in enumerate('xyz')]

        # The PointCloud2 message also has a header which specifies which 
        # coordinate frame it is represented in. 
        header = std_msgs.Header(frame_id=parent_frame)

        return sensor_msgs.PointCloud2(
            header=header,
            height=1, 
            width=points.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=(itemsize * 3), # Every point consists of three float32s.
            row_step=(itemsize * 3 * points.shape[0]),
            data=data
        )

def main(args=None):
    rclpy.init(args=args)
    pcd = PCD()
    pcd.run()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
