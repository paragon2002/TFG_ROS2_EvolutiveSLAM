import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import os

class PublicadorMapa(Node):
    def __init__(self):
        super().__init__('publicador_mapa')
        # Publicamos en el topic que lee tu RViz
        self.publisher_ = self.create_publisher(PointCloud2, '/mapa_evolutivo', 10)
        
        # Ruta al mapa generado por el TFG
        self.ply_path = os.path.expanduser('~/ros2_ws/src/ROS2_Evolutive_Localization/tools/dataset_eficiente/mapa_generado_tfg.ply')
        
        # Timer para leer el disco y publicar cada 10 segundos
        self.timer = self.create_timer(10.0, self.timer_callback)
        print(f"📡 Publicador iniciado. Buscando mapa en: {self.ply_path}")
        print("Enviando datos al topic /mapa_evolutivo cada 10 segundos...")

    def timer_callback(self):
        if not os.path.exists(self.ply_path):
            self.get_logger().warn(f"Esperando a que se genere el mapa en: {self.ply_path} (¿Ya ha terminado la primera nube de evloc_node?)")
            return
            
        try:
            # Leer el archivo PLY con Open3D
            pcd = o3d.io.read_point_cloud(self.ply_path)
            points = np.asarray(pcd.points)
            
            # --- FILTRO DEL SUELO DESACTIVADO ---
            # Si quieres volver a quitar el suelo en el futuro, quita el '#' de la línea de abajo:
            points = points[points[:, 2] > 0.0]
            # ------------------------------------------
            
            if len(points) == 0:
                return
                
            # Crear la cabecera del mensaje ROS2
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = 'odom'  # Marco de referencia síncrono con tu simulación
            
            # Convertir la matriz de puntos a PointCloud2 de ROS2
            msg = pc2.create_cloud_xyz32(header, points)
            
            # Publicar en el topic
            self.publisher_.publish(msg)
            print(f"🔄 [INFO] Mapa actualizado y publicado con {len(points)} puntos.")
            
        except Exception as e:
            self.get_logger().error(f"Error al leer o publicar el mapa: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = PublicadorMapa()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
