import numpy as np
import open3d as o3d
import time
import math
from evloc.common_functions import costfunction3d
from evloc.common_functions import distance_pc_to_point
from evloc.common_functions import spatial_rotation
from evloc.common_classes import Color

class Population:
    """
    An agent of the DE algorithm.
    Includes a position and cost.
    """
    def __init__(self, position=[], cost=[]):
        self.Position = position
        self.Cost = cost
        
    def clone(self):
        return Population(self.Position.copy(), self.Cost.copy())

# --- CAMBIO 1: Añadido el parámetro 'use_odometry' con valor por defecto False ---
def de_6dof(scanCloud, mapCloud, mapmax, mapmin, err_dis, NPini, D, iter_max, F, CR, version_fitness, use_odometry=False):
    
    # =================================================================
    # DEBUG PEDIDO POR EL PROFESOR: CONTEO DE PUNTOS
    # =================================================================
    print("\n--- ANÁLISIS DE CARGA COMPUTACIONAL ---")
    print(f"Puntos del Escáner (Candidato): {len(scanCloud.points)}")
    print(f"Puntos del Mapa (Global): {len(mapCloud.points)}")
    print(f"Comparaciones estimadas por iteración: {len(scanCloud.points) * NPini} búsquedas.")
    print("---------------------------------------\n")
    
    # ⚠️ Descomenta la siguiente línea si quieres ver visualmente las nubes antes de buscar
    # o3d.visualization.draw_geometries([scanCloud, mapCloud], window_name="Nube Local vs Mapa Global")
    # =================================================================


    # =================================================================
    # EL INTERRUPTOR: SLAM PURO vs SLAM HÍBRIDO
    # =================================================================
    if use_odometry:
        print(f"{Color.CYAN}⚙️ Ejecutando en MODO HÍBRIDO (Usando Odometría){Color.END}")
        higherBoundX = mapmax[0]  
        lowerBoundX = mapmin[0]
        higherBoundY = mapmax[1]  
        lowerBoundY = mapmin[1]
        higherBoundZ = mapmax[2]  
        lowerBoundZ = mapmin[2]
        higherAngle_rx = mapmax[3]  
        lowerAngle_rx = mapmin[3]
        higherAngle_ry = mapmax[4]  
        lowerAngle_ry = mapmin[4]
        higherAngle_rz = mapmax[5]  
        lowerAngle_rz = mapmin[5]

        # La "Partícula 0" nace exactamente donde la odometría dice que estamos
        odom_target = (np.array(mapmax) + np.array(mapmin)) / 2.0 
        
    else:
        print(f"{Color.PURPLE}Blind MODO A CIEGAS (SLAM Puro / Pure Scan Matching){Color.END}")
        margen_x = 0.40
        margen_y = 0.40
        margen_theta = 0.60
        
        higherBoundX = margen_x
        lowerBoundX = -margen_x
        higherBoundY = margen_y
        lowerBoundY = -margen_y
        higherBoundZ = 0.01  
        lowerBoundZ = -0.01
        higherAngle_rx = 0.001
        lowerAngle_rx = -0.001
        higherAngle_ry = 0.001
        lowerAngle_ry = -0.001
        higherAngle_rz = margen_theta
        lowerAngle_rz = -margen_theta

        # La "Partícula 0" nace pensando que el robot está totalmente quieto (0,0,0)
        odom_target = np.zeros(6) 
    # =================================================================

    all_best_solutions = [] 

    nVar = D            
    VarSize = [1, nVar]   
    minIt = 50  
    stringcondition = "Max iterations reached"

    empty_population = Population()
    population = [empty_population.clone() for _ in range(NPini)]
    rndmember = np.zeros(6)
    count = 0
    vis1 = 1
    vis2 = 1
    best_particle_cost = 100000000
    worst_particle_cost = 100000
    count_bestfix = 0  
    count_worstfix = 0
    count_avgfix = 0
    ind_reparto_error = 100000
    NP = NPini

    ########## LOOP 1 ###########
    for current_iteration in range(NPini):
        if current_iteration == 0:  
            population[current_iteration].Position = odom_target.copy()
        else:
            rndmember = np.zeros(6)
            for n in range(nVar):
                if n == 0: rndmember[n] = np.random.uniform(lowerBoundX, higherBoundX)
                elif n == 1: rndmember[n] = np.random.uniform(lowerBoundY, higherBoundY)
                elif n == 2: rndmember[n] = np.random.uniform(lowerBoundZ, higherBoundZ)
                elif n == 3: rndmember[n] = np.random.uniform(lowerAngle_rx, higherAngle_rx)
                elif n == 4: rndmember[n] = np.random.uniform(lowerAngle_ry, higherAngle_ry)
                elif n == 5: rndmember[n] = np.random.uniform(lowerAngle_rz, higherAngle_rz)
            population[current_iteration].Position = rndmember

        cand_scan = o3d.geometry.PointCloud()
        cand_scan.points = o3d.utility.Vector3dVector(spatial_rotation(scanCloud.points, population[current_iteration].Position))

        aabb = cand_scan.get_axis_aligned_bounding_box()
        cotout_region = o3d.geometry.AxisAlignedBoundingBox(min_bound=aabb.get_min_bound(), max_bound=aabb.get_max_bound())
        Mapa_3D_cut = mapCloud.crop(cotout_region)

        if Mapa_3D_cut.is_empty():
            Mapa_3D_cut = o3d.geometry.PointCloud()
            Mapa_3D_cut.points = o3d.utility.Vector3dVector(np.zeros((np.array(cand_scan.points).shape[0], 3)))

        kdtree = o3d.geometry.KDTreeFlann(Mapa_3D_cut)
        points_array = np.asarray(Mapa_3D_cut.points)
        
        valid_scan = []
        valid_map = []

        for i in range(len(cand_scan.points)):
            query = np.array(cand_scan.points[i])
            query = np.where(np.isnan(query), 0, query) 
            k, idx, d2 = kdtree.search_knn_vector_3d(query, 1)
            
            # --- FILTRO TRIMMED ICP: Si la distancia es < 0.25 (50cm), lo aceptamos ---
            if d2[0] < 0.25:
                valid_scan.append(cand_scan.points[i])
                valid_map.append(points_array[idx[0]])

        if len(valid_scan) > 20:
            dist_NNmap = distance_pc_to_point(np.array(valid_map), population[current_iteration].Position)
            dist_scancand = distance_pc_to_point(np.array(valid_scan), population[current_iteration].Position)
            base_cost = costfunction3d(dist_scancand, dist_NNmap, version_fitness, err_dis)
        else:
            base_cost = 999999.0 # Penalización severa si no hay solapamiento

        population[current_iteration].Cost = base_cost

    ###### END LOOP 1 ######

    for it in range(iter_max):
        start_time = time.time()
        for pop_id in range(NP-1):
            a, b, c = np.random.randint(0, NP, size=3)
            newmember = Population()
            newmember.Position = [0,0,0,0,0,0]

            for j in range(nVar): 
                if np.random.rand() < CR:
                    newmember.Position[j] = population[c].Position[j] + F * (population[a].Position[j] - population[b].Position[j])
                else:
                    newmember.Position[j] = population[pop_id].Position[j]

            newmember.Position[0] = np.clip(newmember.Position[0], lowerBoundX, higherBoundX)
            newmember.Position[1] = np.clip(newmember.Position[1], lowerBoundY, higherBoundY)
            newmember.Position[2] = np.clip(newmember.Position[2], lowerBoundZ, higherBoundZ)
            newmember.Position[3] = np.clip(newmember.Position[3], lowerAngle_rx, higherAngle_rx)
            newmember.Position[4] = np.clip(newmember.Position[4], lowerAngle_ry, higherAngle_ry)
            newmember.Position[5] = np.clip(newmember.Position[5], lowerAngle_rz, higherAngle_rz)

            cand_scan = o3d.geometry.PointCloud()
            cand_scan.points = o3d.utility.Vector3dVector(spatial_rotation(scanCloud.points, newmember.Position))

            aabb = cand_scan.get_axis_aligned_bounding_box()
            cotout_region = o3d.geometry.AxisAlignedBoundingBox(min_bound=aabb.get_min_bound(), max_bound=aabb.get_max_bound())
            Mapa_3D_cut = mapCloud.crop(cotout_region)

            if Mapa_3D_cut.is_empty():
                Mapa_3D_cut = o3d.geometry.PointCloud()
                Mapa_3D_cut.points = o3d.utility.Vector3dVector(np.zeros((np.array(cand_scan.points).shape[0], 3)))

            kdtree = o3d.geometry.KDTreeFlann(Mapa_3D_cut)
            points_array = np.asarray(Mapa_3D_cut.points)
            
            valid_scan = []
            valid_map = []

            for i in range(len(cand_scan.points)):
                query = np.array(cand_scan.points[i])
                query = np.where(np.isnan(query), 0, query) 
                k, idx, d2 = kdtree.search_knn_vector_3d(query, 1)
                
                # --- FILTRO TRIMMED ICP: Ignorar puntos sin correspondencia cercana ---
                if d2[0] < 0.25:
                    valid_scan.append(cand_scan.points[i])
                    valid_map.append(points_array[idx[0]])

            if len(valid_scan) > 20:
                dist_NNmap = distance_pc_to_point(np.array(valid_map), newmember.Position)
                dist_scancand = distance_pc_to_point(np.array(valid_scan), newmember.Position)
                base_cost = costfunction3d(dist_scancand, dist_NNmap, version_fitness, err_dis)
            else:
                base_cost = 999999.0

            newmember.Cost = base_cost

            if newmember.Cost < population[pop_id].Cost * 0.98:  
                population[pop_id] = newmember

        disc_range=0.9 
        repl_range=0.2 
        population.sort(key=lambda x: x.Cost)
        for i in range(NP, int(disc_range * NP), -1):
            population[i-1] = population[np.random.randint(0, repl_range*NP)]
        population = sorted(population, key=lambda obj: obj.Cost)

        BestSol  = population[0]
        WorstSol = population[NP-1]
        sumcosts=0

        for k in range(NP):
            sumcosts += population[k].Cost
        average_cost = sumcosts / NP

        best_particle_cost_now = BestSol.Cost
        worst_particle_cost_now = WorstSol.Cost

        if count == 10:
            print(f"\nIt: {it}, {Color.GREEN}Best: {round(best_particle_cost_now, 4)}{Color.END}, {Color.RED}Worst: {round(worst_particle_cost_now,4)}{Color.END}, {Color.YELLOW}Average: {round(average_cost,4)}{Color.END}")
            count=0
        count=count+1
        
        if best_particle_cost_now < best_particle_cost:
            count_worstfix = 0; count_avgfix = 0; count_bestfix = 0  
        else:
            count_bestfix += 1  

        best_particle_cost = best_particle_cost_now

        if worst_particle_cost_now > worst_particle_cost:
            count_worstfix = 0; count_avgfix = 0; count_bestfix = 0  
        else:
            count_worstfix += 1  

        worst_particle_cost = worst_particle_cost_now
        ind_reparto_error_aux = sumcosts / (NP*best_particle_cost)

        if (ind_reparto_error_aux < ind_reparto_error): 
            count_avgfix=0; count_worstfix=0; count_bestfix=0  
        else:
            count_avgfix += 1 

        ind_reparto_error = ind_reparto_error_aux

        if (worst_particle_cost/best_particle_cost < 2.5) and (ind_reparto_error < 2): 
            F = 0.7
        if (worst_particle_cost/best_particle_cost < 1.5) and (ind_reparto_error < 1.25): 
            F = 0.3
            NP = int(NPini/5)

        if all(obj.Cost == best_particle_cost_now for obj in population) or \
                (worst_particle_cost / best_particle_cost < 1.15 and ind_reparto_error < 1.15 and it >= minIt) or \
                (count_bestfix > 10 and count_worstfix > 10 and count_avgfix > 10 and it >= minIt):
            
            if all(obj.Cost == best_particle_cost for obj in population):
                stringcondition = 'total convergence'
            elif worst_particle_cost / best_particle_cost < (1.15 + err_dis) and ind_reparto_error < (1.15 + err_dis):
                stringcondition = 'normal convergence'
            else:
                stringcondition = 'invariant convergence'
            
            print(f'\n{Color.CYAN}Population converged in: {it} iterations and condition: {stringcondition}{Color.END}')
            break

        all_best_solutions.append(BestSol.Position)
        
    rmse_array =  BestSol.Cost
    bestCost = BestSol.Cost

    # =================================================================
    # --- CHIVATO: ¿Ganó la odometría o ganó la evolución? ---
    dist_a_particula_cero = np.linalg.norm(np.array(BestSol.Position) - np.array(odom_target))
    
    print("\n--- ANÁLISIS DEL RESULTADO ---")
    if dist_a_particula_cero < 1e-4:
        print(f"{Color.GREEN}¡LA ODOMETRÍA GANÓ! El láser confirma que las ruedas no patinaron (Partícula 0 intacta).{Color.END}")
    else:
        print(f"{Color.YELLOW}¡LA EVOLUCIÓN GANÓ! El láser corrigió la odometría desplazándose {round(dist_a_particula_cero, 4)} unidades de la Partícula 0.{Color.END}")
    print("------------------------------\n")
    # =================================================================

    pcAligned = o3d.geometry.PointCloud()
    pcAligned.points = o3d.utility.Vector3dVector(spatial_rotation(scanCloud.points, all_best_solutions[-1]))

    return(pcAligned, all_best_solutions, bestCost, rmse_array, it, stringcondition)