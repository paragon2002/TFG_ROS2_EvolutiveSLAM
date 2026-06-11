import rclpy 
from rclpy.node import Node
import numpy as np
import open3d as o3d
import csv
import os
import math
import copy
import warnings
from ament_index_python.packages import get_package_share_directory
from evloc.common_classes import Color
from evloc.generate_point_cloud import generate_point_cloud
from evloc.ask_params import ask_params

warnings.filterwarnings("ignore")

def configure_environment():
    return {"online": False, "environment_type": 2, "package_path": os.path.join(get_package_share_directory('evloc'), 'resources', 'simulation')}

def quat_to_euler(qx, qy, qz, qw):
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    sinp = 2 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    return [roll, pitch, yaw]

def create_transform(x, y, z, roll, pitch, yaw):
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[0, 3] = x; T[1, 3] = y; T[2, 3] = z
    return T

def get_relative_jump(odom_prev, odom_act):
    T_prev = create_transform(*odom_prev)
    T_act = create_transform(*odom_act)
    T_rel = np.linalg.inv(T_prev) @ T_act
    x = T_rel[0, 3]
    y = T_rel[1, 3]
    z = T_rel[2, 3]
    sy = math.sqrt(T_rel[0,0]**2 + T_rel[1,0]**2)
    if sy > 1e-6:
        roll = math.atan2(T_rel[2,1], T_rel[2,2])
        pitch = math.atan2(-T_rel[2,0], sy)
        yaw = math.atan2(T_rel[1,0], T_rel[0,0])
    else:
        roll = math.atan2(-T_rel[1,2], T_rel[1,1])
        pitch = math.atan2(-T_rel[2,0], sy)
        yaw = 0
    return np.array([x, y, z, roll, pitch, yaw])

class PCD(Node):
    def __init__(self,config):
        super().__init__('pcd_node')
        # --- CAMBIO APLICADO AQUÍ ---
        self.DATASET_FOLDER = os.path.join(os.path.expanduser('~'), 'ros2_ws', 'src', 'ROS2_Evolutive_Localization', 'tools', 'dataset_eficiente')
        self.GROUNDTRUTH_FILE_PATH= os.path.join(self.DATASET_FOLDER, 'odometria.csv')
        self.mapa_completo = o3d.geometry.PointCloud()
        self.matriz_global = np.eye(4)
        
        # =================================================================
        # PANEL DE CONTROL PARA LAS PRUEBAS DEL TFG
        # =================================================================
        self.MODO_HIBRIDO = True       # False = SLAM Puro (a ciegas) | True = SLAM Híbrido (usa odometría)
        self.INYECTAR_RUIDO = True    # True = Mete un error artificial a la odometría para probar su robustez
        self.VOXEL_SIZE_LOCAL = 0.10   # 0.10 = Reduce puntos cada 10cm. (Pon 0.0 si no quieres reducir la nube)
        # =================================================================

    def run(self):
        start_cloud, err_dis, unif_noise, alg_type, fit_ver, np_ini, iter_max, MAX_CLOUD_ID = ask_params(local_clouds_folder=self.DATASET_FOLDER, online=False)

        odom_inicio = self.leer_fila_excel(start_cloud - 1)
        self.matriz_global = create_transform(*odom_inicio)
        
        nube_1 = o3d.io.read_point_cloud(os.path.join(self.DATASET_FOLDER, f"cloud_{start_cloud-1}.ply"))
        puntos_base = np.asarray(nube_1.points)
        p_h_base = np.hstack((puntos_base, np.ones((puntos_base.shape[0], 1))))
        p_g_base = (self.matriz_global @ p_h_base.T).T[:, :3]
        self.mapa_completo.points = o3d.utility.Vector3dVector(p_g_base)
        
        print(f"\n{Color.YELLOW}🚀 MAPA INICIADO. Nube {start_cloud-1} fijada como ancla.{Color.END}")

        for id_cloud in range(start_cloud, MAX_CLOUD_ID + 1):
            print(f"\n{Color.DARKCYAN}================ PROCESANDO NUBE {id_cloud} CONTRA EL MAPA ================={Color.END}")

            map_local_ori = o3d.io.read_point_cloud(os.path.join(self.DATASET_FOLDER, f"cloud_{id_cloud}.ply"))
            
            # --- PUNTO 1: DOWNSAMPLING DE LA NUBE LOCAL (Acelera el cálculo) ---
            if self.VOXEL_SIZE_LOCAL > 0.0:
                puntos_originales = len(map_local_ori.points)
                map_local_ori = map_local_ori.voxel_down_sample(voxel_size=self.VOXEL_SIZE_LOCAL)
                print(f"{Color.YELLOW}📉 Downsampling Local: Nube reducida de {puntos_originales} a {len(map_local_ori.points)} puntos.{Color.END}")
            # --------------------------------------------------------------------

            odom_act = self.leer_fila_excel(id_cloud)
            if np.all(odom_act == 0):
                print(f"{Color.RED}🚨 ERROR: Nube {id_cloud} no encontrada en odometria.csv{Color.END}")

            odom_prev = self.leer_fila_excel(id_cloud-1)
            desp = get_relative_jump(odom_prev, odom_act)
            
            # --- PUNTO 2 INYECCIÓN DE RUIDO A LA ODOMETRÍA ---
            if self.MODO_HIBRIDO and self.INYECTAR_RUIDO:
                # Simulamos un derrape de las ruedas inyectando un error extra de entre 5cm y 10cm en los ejes
                ruido_x = np.random.choice([-1, 1]) * np.random.uniform(0.75, 1.00) 
                ruido_y = np.random.choice([-1, 1]) * np.random.uniform(0.75, 1.00)
                ruido_yaw = np.random.choice([-1, 1]) * np.random.uniform(0.2, 0.6)
                
                desp_original = copy.deepcopy(desp)
                desp[0] += ruido_x
                desp[1] += ruido_y
                desp[5] += ruido_yaw
                print(f"{Color.RED}⚠️ RUIDO INYECTADO: Las ruedas dicen que avanzó {desp[0]:.3f}m, pero lo real era {desp_original[0]:.3f}m{Color.END}")
            # ----------------------------------------------------------------
            
            margen_tfg = np.array([1.20, 1.20, 0.0, 0.0, 0.0, 0.8])
            print(f"{Color.CYAN}⚙️ Optimizando (Scan-to-Map) con margen {math.degrees(margen_tfg[5]):.2f}º...{Color.END}")

            mapa_perspectiva_local = copy.deepcopy(self.mapa_completo)
            mapa_perspectiva_local.transform(np.linalg.inv(self.matriz_global))

            # --- PUNTO 3: PASAMOS EL INTERRUPTOR use_odometry AL ALGORITMO ---
            # Aquí es donde le dices la población que quieres usar en cada prueba cambiando user_NPini
            all_best_solutions = generate_point_cloud(auto=True, id_cloud=id_cloud, err_dis=err_dis, 
                                          unif_noise=unif_noise, algorithm_type=alg_type, version_fitness=fit_ver, 
                                          user_NPini=200, 
                                          user_iter_max=150, 
                                          map_global=mapa_perspectiva_local, 
                                          real_scan=map_local_ori, groundtruth=desp, 
                                          mapmax=(desp+margen_tfg), mapmin=(desp-margen_tfg),
                                          use_odometry=self.MODO_HIBRIDO) # <--- Variable añadida
            # ------------------------------------------------------------------

            res_final = all_best_solutions[-1]
            x_r, y_r, z_r, ro_r, pi_r, ya_r = res_final
            
            T_paso = create_transform(x_r, y_r, z_r, ro_r, pi_r, ya_r)
            self.matriz_global = self.matriz_global @ T_paso

            puntos_nube = np.asarray(map_local_ori.points)
            puntos_homogeneos = np.hstack((puntos_nube, np.ones((puntos_nube.shape[0], 1))))
            puntos_globales = (self.matriz_global @ puntos_homogeneos.T).T[:, :3]
            nube_transformada = o3d.geometry.PointCloud()
            nube_transformada.points = o3d.utility.Vector3dVector(puntos_globales)
            
            self.mapa_completo += nube_transformada
            self.mapa_completo = self.mapa_completo.voxel_down_sample(voxel_size=0.05)
            
            o3d.io.write_point_cloud(os.path.join(self.DATASET_FOLDER, "mapa_generado_tfg.ply"), self.mapa_completo)
            print(f"{Color.GREEN}✅ Nube {id_cloud} integrada firmemente en el mapa global.{Color.END}")

    def leer_fila_excel(self, id_c):
        try:
            with open(self.GROUNDTRUTH_FILE_PATH, 'r') as file:
                for row in csv.reader(file):
                    try:
                        if float(row[0]) == float(id_c):
                            return np.array([float(row[1]), float(row[2]), float(row[3]), *quat_to_euler(float(row[4]), float(row[5]), float(row[6]), float(row[7]))])
                    except: pass
        except: pass
        return np.zeros(6)

def main(args=None):
    rclpy.init(args=args)
    pcd = PCD(configure_environment())
    pcd.run()
    rclpy.shutdown()

if __name__ == '__main__':
    main()