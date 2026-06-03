import numpy as np
import open3d as o3d
import csv
import os
import math

def quat_to_euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return [roll, pitch, yaw]

DATASET_FOLDER = os.path.expanduser("~/ros2_ws/src/ROS2_Evolutive_Localization/tools/dataset_real/")
EXCEL_PATH = os.path.join(DATASET_FOLDER, "odometria.csv")
MAPA_SALIDA = os.path.join(DATASET_FOLDER, "mapa_SOLO_odometria.ply")

def leer_fila_excel(id_c):
    with open(EXCEL_PATH, 'r') as file:
        for row in csv.reader(file):
            try:
                if float(row[0]) == float(id_c):
                    angulos = quat_to_euler(float(row[4]), float(row[5]), float(row[6]), float(row[7]))
                    return np.array([float(row[1]), float(row[2]), float(row[3]), angulos[0], angulos[1], angulos[2]])
            except:
                pass
    return None

def spatial_rotation(points, position):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    R = pcd.get_rotation_matrix_from_xyz((position[3], position[4], position[5]))
    pcd.rotate(R, center=(0, 0, 0))
    pcd.translate((position[0], position[1], position[2]))
    return np.asarray(pcd.points)

def main():
    mapa_odometria = o3d.geometry.PointCloud()
    print("Empezando a generar mapa SOLO con la Odometría del Excel...")
    
    id_cloud = 1
    while True:
        ruta_nube = os.path.join(DATASET_FOLDER, f"cloud_{id_cloud}.ply")
        if not os.path.exists(ruta_nube):
            print(f"\nNo se encontraron más nubes. Última nube procesada: {id_cloud - 1}")
            break
            
        nube_local = o3d.io.read_point_cloud(ruta_nube)
        odom_absoluta = leer_fila_excel(id_cloud)
        
        if odom_absoluta is not None:
            puntos_rotados = spatial_rotation(np.asarray(nube_local.points), odom_absoluta)
            nube_transformada = o3d.geometry.PointCloud()
            nube_transformada.points = o3d.utility.Vector3dVector(puntos_rotados)
            
            mapa_odometria += nube_transformada
            print(f"Nube {id_cloud} integrada correctamente.")
            
        id_cloud += 1
            
    mapa_odometria = mapa_odometria.voxel_down_sample(voxel_size=0.03)
    o3d.io.write_point_cloud(MAPA_SALIDA, mapa_odometria)
    print(f"¡Éxito! El mapa del Excel se ha guardado en: {MAPA_SALIDA}")

if __name__ == '__main__':
    main()
