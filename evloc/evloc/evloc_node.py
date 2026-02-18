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

def configure_environment():
    # Preguntar si es online
    online_input = input("\nWork in online mode? (y/n): ")

    if online_input.lower() == 'y':
        online = True
    else:
        online = False

    # Preguntar tipo de environment
    print("\nSelect Environment:")
    print(" 1) Real Environment (Default)")
    print(" 2) Simulation Dataset")

    environment_type = input()

    if not environment_type.strip():
        environment_type = 1
        print('\t Option 1 (Real) by default.')
    else:
        environment_type = int(environment_type)

    env_paths = {
        1: 'real',
        2: 'simulation'
    }

    BASE_PACKAGE_PATH = os.path.join(get_package_share_directory('evloc'))

    PACKAGE_PATH = os.path.join(
        BASE_PACKAGE_PATH,
        'resources',
        env_paths[environment_type]
    )

    config = {
        "online": online,
        "environment_type": environment_type,
        "package_path": PACKAGE_PATH
    }

    return config

# Filter out the RuntimeWarning for invalid value encountered in divide
# For when NaN is calculated in add_noise_to_pc.
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

##############################################################

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

    def __init__(self,config):
        super().__init__('pcd_node')
        
        # Declara el parámetro mi_parametro con un valor predeterminado
        self.declare_parameter('auto', False)
        self.online = config["online"]
        self.environment_type = config["environment_type"]
        self.PACKAGE_PATH = config["package_path"]
        self.declare_parameter('animation', False)

        auto_color = Color.RED
        online_color = Color.RED
        animation_color = Color.RED

        self.auto_mode = self.get_parameter('auto').value
        # self.online = self.get_parameter('online').value
        self.animation = self.get_parameter('animation').value

        if self.auto_mode:
            auto_color = Color.GREEN

        if self.online:
            online_color = Color.GREEN

        if self.animation:
            animation_color = Color.GREEN

        print(auto_color + f"\nAuto Mode: {self.auto_mode}" + Color.END)
        print(online_color + f"online: {self.online}" + Color.END)
        print(animation_color + f"Animation: {self.animation}" + Color.END)

        # Configuración según environment_type
        if self.environment_type == 1:
            self.DOWN_SAMPLING_FACTOR_GLOBAL = 250
            self.DOWN_SAMPLING_FACTOR = 100 # 0.01
            self.global_downsample_show = 1
            self.map_global_ori= o3d.io.read_point_cloud(
                f"{self.PACKAGE_PATH}/map_global_ori.ply"
            )
        elif self.environment_type == 2:
            self.DOWN_SAMPLING_FACTOR_GLOBAL = 1 #320
            self.DOWN_SAMPLING_FACTOR = 1
            self.global_downsample_show = 1
            self.map_global_ori = o3d.io.read_point_cloud(
                f"{self.PACKAGE_PATH}/map_global.pcd"
            )
            self.map_global_ori = filter_map_height(self.map_global_ori, 0, 1.35)

        self.GROUNDTRUTH_FILE_PATH= f"{self.PACKAGE_PATH}/groundtruth_data.csv"

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

        self.pcd_publisher_local = self.create_publisher(sensor_msgs.PointCloud2, 'evloc_local', 10)
        self.pcd_publisher_global = self.create_publisher(sensor_msgs.PointCloud2, 'evloc_global', 10)
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

    def run(self):

        fixed_frame = 'base_footprint'

        if (self.online):
            rclpy.spin_once(self) # Read once from subscribed topics
            while (self.cloud_points == None or np.all(np.isinf(self.groundtruth))):
                print("Waiting for local scan...")
                rclpy.spin_once(self)
            
            map_global_unfiltered = o3d.io.read_point_cloud(f"{PACKAGE_PATH}/map_global.pcd")
            map_global = filter_map_height(map_global_unfiltered, 0, 1.35)
            global_downsample = 1

        else:
            self.map_global = self.map_global_ori.uniform_down_sample(every_k_points=int(self.DOWN_SAMPLING_FACTOR_GLOBAL)) # Original PointCloud (Global Map)
            


            
        
        points2 = np.asarray(self.map_global.points)
        print(f"Global Map size: {points2.shape} .")
        map_global_show = filter_map_height(self.map_global_ori, 0, 1.35)
        points2 = np.asarray(map_global_show.points)[::self.global_downsample_show] # Downsampling. Son demasiados puntos para RVIZ
        pcd_global = self.point_cloud(points2, fixed_frame)
        self.pcd_publisher_global.publish(pcd_global)
        print(f"Global PointCloud with dimensions {points2.shape} has been published.")

        # Ask once before starting if in auto mode.
        LOCAL_CLOUDS_FOLDER = os.path.join(self.PACKAGE_PATH, "local_clouds")
        if self.auto_mode:
            
            id_cloud, err_dis, unif_noise, algorithm_type, version_fitness, user_NPini, user_iter_max, MAX_CLOUD_ID = ask_params(local_clouds_folder=LOCAL_CLOUDS_FOLDER)
            
            
        while True:

            print(Color.BOLD + "\n------------------------------------" + Color.END)

            # Ask every iteration if not in auto mode.
            if not self.auto_mode:
               
               id_cloud, err_dis, unif_noise, algorithm_type, version_fitness, user_NPini, user_iter_max, MAX_CLOUD_ID  = ask_params(local_clouds_folder=LOCAL_CLOUDS_FOLDER)

            map_local = None
            real_groundtruth = None

            if (self.online):
                rclpy.spin_once(self) # Read once from subscribed topics
                while (self.cloud_points == None or np.all(np.isinf(self.groundtruth))):
                    print("Waiting for local scan...")
                    rclpy.spin_once(self)

                real_groundtruth = self.groundtruth

                # Transform map_local datatype
                points = read_points(self.cloud_points, skip_nans=True, field_names=("x", "y", "z"))
                point_list = np.array(list(points))
                map_local_unfiltered = o3d.geometry.PointCloud()
                map_local_unfiltered.points = o3d.utility.Vector3dVector(point_list)
                map_local = filter_map_height(map_local_unfiltered, 0, 1.35)

            else:
                real_scan_ori = o3d.io.read_point_cloud(f"{LOCAL_CLOUDS_FOLDER}/cloud_{id_cloud}.ply")
                map_local = real_scan_ori.uniform_down_sample(every_k_points=int(self.DOWN_SAMPLING_FACTOR)) # User Selected PointCloud (Local Map)
                real_groundtruth = get_groundtruth_data(self.GROUNDTRUTH_FILE_PATH, id_cloud)              

            print(f"Obtained local scan with dimensions {np.asarray(map_local.points).shape}\n")

            # all_best_solutions is a list containing the best solution found each iteration of the algorithm.
            # The last element of the list will be the best solution of them all.
            all_best_solutions = generate_point_cloud(auto=self.auto_mode,
                                          id_cloud = id_cloud,
                                          err_dis = err_dis, 
                                          unif_noise = unif_noise,
                                          algorithm_type = algorithm_type,
                                          version_fitness = version_fitness,
                                          user_NPini = user_NPini,
                                          user_iter_max = user_iter_max,
                                          map_global = self.map_global,
                                          real_scan = map_local,
                                          groundtruth = real_groundtruth)


            if self.animation:
                count = 0
                animation_not_finished = self.ask_restart("Start Animation? (y/n): ")
                while animation_not_finished:
                    for sol in all_best_solutions:
                        count += 1

                        points = spatial_rotation(map_local.points, sol)

                        if points is None:
                            print("Error generating point cloud.")
                            break
                        
                        ds_1 = 1
                        if not self.online:
                            ds_1 = 1
                        
                        self.publish_point_clouds(points, fixed_frame, ds_1, silent=True)
                        print(f"{Color.BOLD} Published solution {count}/{len(all_best_solutions)} {Color.END}")
                        time.sleep(1)

                    print(f"\n{Color.BOLD} Animation Finished {Color.END}")
                    count = 0
                    animation_not_finished = self.ask_restart("Restart Animation? (y/n): ")

            else:
                # Just show the best solution of them all. (last element of all_best_solutions)
                points = spatial_rotation(map_local.points, all_best_solutions[-1])

                if points is None:
                    print("Error generating point cloud.")
                    break
                
                # Por si se quiere hacer downsampling antes de publicar
                ds_1 = 1
                if not self.online:
                    ds_1 = 1
                
                self.publish_point_clouds(points, 'base_footprint', ds_1)

            # Reset variables obtained from simulation
            self.cloud_points = None
            self.groundtruth = np.full(6, np.inf)

            if not self.auto_mode:
                restart = self.ask_restart("Restart node? (y/n): ")
                if not restart:
                    self.destroy_node()  # Cierra el nodo antes de salir del bucle
                    break
            else:
                if not self.online:
                    # Loop for every cloud when in auto mode
                    id_cloud += 1
                    if id_cloud > MAX_CLOUD_ID:
                        id_cloud = MIN_CLOUD_ID

            print(Color.BOLD + "\n------------------------------------" + Color.END)

    def ask_restart(self, text):
        while True:
            user_input = input(text)
            if user_input == 'y':
                return True
            elif user_input == 'n':
                return False
            else:
                print("Invalid answer. Please type 'y' for yes or 'n' for no.")


    def publish_point_clouds(self, points, parent_frame, downsample_1, silent=False):
        
        points = points[::downsample_1]
        pcd = self.point_cloud(points, parent_frame)
        self.pcd_publisher_local.publish(pcd)
        if not silent:
            print(f"Local PointCloud with dimensions {points.shape} has been published.")

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
    config = configure_environment()
    rclpy.init(args=args)
    pcd = PCD(config)
    pcd.run()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
