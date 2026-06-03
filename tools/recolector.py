import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import csv
import threading
import os
import sys

class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')
        
        # Nos suscribimos a la odometría y al láser 3D
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.sub_pc = self.create_subscription(PointCloud2, '/velodyne_points', self.pc_cb, 10)

        self.latest_odom = None
        self.latest_pc = None
        self.save_counter = 1

        # Crear carpeta para guardar los datos de forma segura
        self.save_dir = "dataset_real"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Preparar el archivo CSV para la odometría
        self.csv_path = os.path.join(self.save_dir, 'odometria.csv')
        # Si el archivo no existe, le ponemos las cabeceras
        es_nuevo = not os.path.exists(self.csv_path)
        self.csv_file = open(self.csv_path, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        if es_nuevo:
            self.csv_writer.writerow(['id_nube', 'pos_x', 'pos_y', 'pos_z', 'ori_x', 'ori_y', 'ori_z', 'ori_w'])

    def odom_cb(self, msg):
        self.latest_odom = msg

    def pc_cb(self, msg):
        self.latest_pc = msg

    def save_data(self):
        if self.latest_odom is None:
            print("⚠️ Esperando datos de Odometría (/odom)... Mueve un poco el robot.")
            return
        if self.latest_pc is None:
            print("⚠️ Esperando datos del Láser (/velodyne_points)...")
            return

        # 1. Guardar Odometría en el CSV
        odom = self.latest_odom
        pos = odom.pose.pose.position
        ori = odom.pose.pose.orientation
        self.csv_writer.writerow([self.save_counter, pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w])
        self.csv_file.flush() # Forzar el guardado en el disco duro

        # 2. Guardar Nube de Puntos en .PLY
        points = []
        for p in pc2.read_points(self.latest_pc, field_names=("x", "y", "z"), skip_nans=True):
            points.append([p[0], p[1], p[2]])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        filename = os.path.join(self.save_dir, f"cloud_{self.save_counter}.ply")
        o3d.io.write_point_cloud(filename, pcd)

        print(f"✅ [{self.save_counter}] ¡Guardado perfecto! -> {filename} y fila en odometria.csv")
        self.save_counter += 1

def main(args=None):
    rclpy.init(args=args)
    node = DataCollector()

    # Ponemos ROS a escuchar en un hilo secundario para no bloquear la terminal
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    print("\n" + "="*50)
    print("🚀 RECOLECTOR DE DATOS INICIADO 🚀")
    print("="*50)
    print("Instrucciones:")
    print("1. Conduce el robot con tu teclado en otra terminal.")
    print("2. Vuelve a ESTA terminal y presiona la tecla ENTER para guardar una captura.")
    print("3. Escribe 'q' y presiona ENTER para salir de forma segura.")
    print("="*50 + "\n")

    try:
        while rclpy.ok():
            cmd = input("Presiona [ENTER] para capturar (o 'q' para salir): ")
            if cmd.lower() == 'q':
                break
            node.save_data()
    except KeyboardInterrupt:
        pass
    finally:
        node.csv_file.close()
        rclpy.shutdown()
        thread.join()

if __name__ == '__main__':
    main()
