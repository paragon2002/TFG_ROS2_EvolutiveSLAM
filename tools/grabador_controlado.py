import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import open3d as o3d
import math
import os
import csv

class GrabadorNubes(Node):
    def __init__(self):
        super().__init__('grabador_controlado')
        
        # --- CONFIGURACIÓN DE SALTOS ---
        self.dist_th = 0.20  # 20 centímetros
        self.angle_th = math.radians(20.0)  # 20 grados pasados a radianes
        
        # --- CONFIGURACIÓN DE TÓPICOS ---
        self.TOPICO_NUBES = '/velodyne_points'  
        self.TOPICO_ODOM = '/odom'             
        
        # Crear carpeta nueva
        # --- CAMBIO REALIZADO AQUÍ: dataset_eficiente ---
        self.folder = os.path.expanduser('~/ros2_ws/src/ROS2_Evolutive_Localization/tools/dataset_eficiente')
        os.makedirs(self.folder, exist_ok=True)
        self.csv_path = os.path.join(self.folder, 'odometria.csv')

        # Inicializar y limpiar el CSV
        with open(self.csv_path, 'w', newline='') as f:
            pass 

        self.last_pose = None
        self.latest_cloud = None
        self.cloud_count = 1

        # Suscriptores
        self.create_subscription(PointCloud2, self.TOPICO_NUBES, self.cloud_cb, 10)
        self.create_subscription(Odometry, self.TOPICO_ODOM, self.odom_cb, 10)
        
        print(f"✅ Grabador iniciado. Esperando datos en {self.TOPICO_NUBES} y {self.TOPICO_ODOM}...")

    def cloud_cb(self, msg):
        # Guarda siempre en memoria la última nube que ve el láser
        self.latest_cloud = msg

    def get_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_cb(self, msg):
        if self.latest_cloud is None:
            return

        # Sacar posición actual
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = self.get_yaw(msg.pose.pose.orientation)
        current_pose = (x, y, yaw)

        # Si es la primera nube, guardarla directamente
        if self.last_pose is None:
            self.save_data(msg.pose.pose, current_pose)
            return

        # Calcular diferencias
        dx = x - self.last_pose[0]
        dy = y - self.last_pose[1]
        dist = math.sqrt(dx**2 + dy**2)
        
        dyaw = abs(yaw - self.last_pose[2])
        if dyaw > math.pi:
            dyaw = 2 * math.pi - dyaw

        # Si supera 20cm o 20 grados -> ¡GUARDAR!
        if dist >= self.dist_th or dyaw >= self.angle_th:
            self.save_data(msg.pose.pose, current_pose)

    def save_data(self, pose_msg, current_pose):
        # Leer puntos asegurando que solo cogemos x, y, z
        puntos = list(pc2.read_points(self.latest_cloud, field_names=("x", "y", "z"), skip_nans=True))
        
        # SEGURO ANTICAÍDAS: Si la lista está vacía, no hacemos nada
        if len(puntos) == 0:
            self.get_logger().warn("Nube vacía ignorada (Gazebo cargando)...")
            return

        # LA MAGIA PARA EVITAR EL ERROR 'NUMPY.VOID':
        # Extraemos manualmente x, y, z y lo forzamos a una matriz de decimales puros (float64)
        points_arr = np.array([ [p[0], p[1], p[2]] for p in puntos ], dtype=np.float64)
        
        pcd = o3d.geometry.PointCloud()
        # Ahora Open3D recibe exactamente lo que quiere: una matriz de Nx3 números puros
        pcd.points = o3d.utility.Vector3dVector(points_arr)
        
        # Guardar .ply
        ply_path = os.path.join(self.folder, f'cloud_{self.cloud_count}.ply')
        o3d.io.write_point_cloud(ply_path, pcd)

        # Guardar fila en CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            # Formato: id, x, y, z, qx, qy, qz, qw
            writer.writerow([self.cloud_count, pose_msg.position.x, pose_msg.position.y, pose_msg.position.z,
                             pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w])
        
        print(f"📸 Nube {self.cloud_count} guardada! (Salto detectado)")
        self.last_pose = current_pose
        self.cloud_count += 1
        self.latest_cloud = None # Resetear

def main(args=None):
    rclpy.init(args=args)
    node = GrabadorNubes()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
