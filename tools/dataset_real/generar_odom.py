import os
import csv
import math
import numpy as np
import open3d as o3d

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

folder = os.path.expanduser('~/ros2_ws/src/ROS2_Evolutive_Localization/tools/dataset_real')
csv_file = os.path.join(folder, 'odometria.csv')

# Leer todo el Excel
odom_dict = {}
with open(csv_file, 'r') as f:
    for row in csv.reader(f):
        try:
            odom_dict[int(float(row[0]))] = np.array([float(row[1]), float(row[2]), float(row[3]), *quat_to_euler(float(row[4]), float(row[5]), float(row[6]), float(row[7]))])
        except: pass

mapa_odom = o3d.geometry.PointCloud()

# Procesar nubes súper rápido (sin algoritmo)
for i in range(1, 1000):
    cloud_path = os.path.join(folder, f"cloud_{i}.ply")
    if not os.path.exists(cloud_path):
        continue # <--- AQUÍ ESTÁ EL CAMBIO. SALTA LA NUBE SI NO EXISTE
    if i not in odom_dict:
        continue
        
    nube = o3d.io.read_point_cloud(cloud_path)
    T = create_transform(*odom_dict[i])
    
    puntos = np.asarray(nube.points)
    puntos_h = np.hstack((puntos, np.ones((len(puntos), 1))))
    puntos_g = (T @ puntos_h.T).T[:, :3]
    
    nube_t = o3d.geometry.PointCloud()
    nube_t.points = o3d.utility.Vector3dVector(puntos_g)
    mapa_odom += nube_t
    print(f"Nube {i} sumada (Odometría pura)")

mapa_odom = mapa_odom.voxel_down_sample(voxel_size=0.03)
o3d.io.write_point_cloud(os.path.join(folder, "mapa_odometria_pura.ply"), mapa_odom)
print("✅ ¡Mapa de odometría guardado como mapa_odometria_pura.ply!")
