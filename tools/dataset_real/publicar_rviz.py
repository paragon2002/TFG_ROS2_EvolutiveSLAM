import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import open3d as o3d
import numpy as np
import os
import sensor_msgs_py.point_cloud2 as pc2

class PublicadorMapa(Node):
    def __init__(self):
        super().__init__('publicador_mapa')
        self.publisher_ = self.create_publisher(PointCloud2, '/mapa_recortado', 10)
        
        # Publicamos cada 5 segundos para no saturar RViz
        self.timer = self.create_timer(5.0, self.timer_callback)
        
        self.ruta = os.path.expanduser("~/ros2_ws/src/ROS2_Evolutive_Localization/tools/dataset_real/mapa_generado_tfg.ply")
        
        # Configuracion del corte de altura
        self.z_min = -1.0
        self.z_max = 5.0

        print(f"📡 Publicador iniciado. Frame: mapa_recortado | Topic: /mapa_recortado")
        print(f"📂 Leyendo de: {self.ruta}")

    def timer_callback(self):
        # 1. Comprobamos si el archivo existe para no dar error
        if not os.path.exists(self.ruta):
            self.get_logger().warn(f"Esperando a que el algoritmo cree el archivo: {self.ruta}")
            return

        try:
            # 2. Leemos el mapa (esto permite que se actualice si el archivo crece)
            pcd = o3d.io.read_point_cloud(self.ruta)
            puntos = np.asarray(pcd.points)

            if len(puntos) == 0:
                return

            # 3. Aplicamos el corte de altura
            mascara = (puntos[:, 2] >= self.z_min) & (puntos[:, 2] <= self.z_max)
            puntos_filtrados = puntos[mascara]

            # 4. Creamos el mensaje para RViz
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = 'mapa_recortado' # <-- IMPORTANTE: Pon esto en el Fixed Frame de RViz
            
            msg = pc2.create_cloud_xyz32(header, puntos_filtrados.tolist())
            
            # 5. Publicamos
            self.publisher_.publish(msg)
            print(f"✅ Mapa actualizado: {len(puntos)} puntos totales | {len(puntos_filtrados)} en pantalla")

        except Exception as e:
            self.get_logger().error(f"Error al leer el mapa: {e}")

def main(args=None):
    rclpy.init(args=args)
    nodo = PublicadorMapa()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    nodo.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
