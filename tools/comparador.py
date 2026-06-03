import open3d as o3d
import numpy as np
import os
import csv

# Importamos la "batidora" matemática de tu profesor
from evloc.generate_point_cloud import generate_point_cloud

def obtener_movimiento_ruedas(ruta_csv, id_nube_1, id_nube_2):
    odom_1, odom_2 = np.zeros(6), np.zeros(6)
    
    with open(ruta_csv, 'r') as f:
        lector = csv.reader(f)
        for fila in lector:
            try:
                # Buscamos la fila cuyo primer número sea el ID de la nube
                if float(fila[0]) == float(id_nube_1):
                    odom_1 = np.array(fila[1:7], dtype=float)
                elif float(fila[0]) == float(id_nube_2):
                    odom_2 = np.array(fila[1:7], dtype=float)
            except ValueError:
                continue # Ignoramos la primera fila si tiene letras (cabecera)
                
    # Restamos para ver cuánto se ha movido según las ruedas
    movimiento = odom_2 - odom_1
    return movimiento

def evaluar_nubes(id_nube_1, id_nube_2):
    print(f"\n--- Iniciando Emparejamiento: Nube {id_nube_2} sobre Nube {id_nube_1} ---")
    
    # 1. Rutas a tu dataset
    carpeta_dataset = os.path.join(os.path.expanduser('~'), 'ros2_ws', 'src', 'ROS2_Evolutive_Localization', 'tools', 'dataset_real')
    ruta_nube_1 = os.path.join(carpeta_dataset, f'cloud_{id_nube_1}.ply')
    ruta_nube_2 = os.path.join(carpeta_dataset, f'cloud_{id_nube_2}.ply')
    ruta_csv = os.path.join(carpeta_dataset, 'odometria.csv')

    # 2. Leer los datos
    print("Cargando fotos 3D y leyendo el archivo Excel (Odometría)...")
    mapa_global = o3d.io.read_point_cloud(ruta_nube_1)
    mapa_local = o3d.io.read_point_cloud(ruta_nube_2)
    movimiento_ruedas = obtener_movimiento_ruedas(ruta_csv, id_nube_1, id_nube_2)

    print(f"Puntos Nube 1: {np.asarray(mapa_global.points).shape}")
    print(f"Puntos Nube 2: {np.asarray(mapa_local.points).shape}")

    # 3. Groundtruth falso para engañar al programa
    dummy_groundtruth = np.zeros(6) 
    
    print("\nLlamando al algoritmo de Evolución Diferencial de tu tutor...")
    # 4. Cálculos
    mejores_soluciones = generate_point_cloud(
        auto=True,                 
        id_cloud=id_nube_2, 
        err_dis=0.0,               
        unif_noise=0.0,            
        algorithm_type=1,          
        version_fitness=1,         
        user_NPini=50,             
        user_iter_max=50,          
        map_global=mapa_global,    
        real_scan=mapa_local,      
        groundtruth=dummy_groundtruth
    )

    resultado_algoritmo = mejores_soluciones[-1]
    
    # 5. La Tabla de Comparación Definitiva
    print("\n" + "="*80)
    print(f"{'RESULTADO FINAL: ALGORITMO vs ODOMETRÍA (RUEDAS)':^80}")
    print("="*80)
    print(f"{'Eje / Giro':<15} | {'Algoritmo (Láser)':<20} | {'Odometría (Ruedas)':<20} | {'Diferencia (Error)':<20}")
    print("-" * 80)
    
    nombres = ['Eje X (m)', 'Eje Y (m)', 'Eje Z (m)', 'Roll (rad)', 'Pitch(rad)', 'Yaw (rad)']
    for i in range(6):
        error = abs(resultado_algoritmo[i] - movimiento_ruedas[i])
        print(f"{nombres[i]:<15} | {resultado_algoritmo[i]:>18.4f}   | {movimiento_ruedas[i]:>18.4f}   | {error:>18.4f}")
    print("="*80 + "\n")

if __name__ == '__main__':
    evaluar_nubes(4, 5)
