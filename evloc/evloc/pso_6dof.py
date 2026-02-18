import numpy as np
import open3d as o3d
import time
from evloc.common_functions import costfunction3d
from evloc.common_functions import distance_pc_to_point
from evloc.common_functions import spatial_rotation
from evloc.common_classes import Color


class Particle:
    """
    Part of the PSO algortihm
    """
    def __init__(self, n_dims):
        self.velocity = np.zeros(n_dims)
        self.position = np.random.uniform(-1, 1, n_dims)
        self.best_position = np.copy(self.position)
        self.cost = float('inf')
        self.best_cost = float('inf')


class Swarm:
    """
    Part of the PSO algortihm
    """
    def __init__(self, n_particles, n_dims):
        self.particles = [Particle(n_dims) for _ in range(n_particles)]
        self.global_best_position = np.zeros(n_dims)
        self.global_best_cost = float('inf')


def pso_6dof(scanCloud,mapCloud,mapmax,mapmin,err_dis,NPini,D, w, wdamp, c1, c2, iter_max, version_fitness):
    ##  Boundaries
    higherBoundX = mapmax[0]  # X Translation in meters
    lowerBoundX = mapmin[0]
    higherBoundY = mapmax[1]  # Y Translation in meters
    lowerBoundY = mapmin[1]
    higherBoundZ = mapmax[2]  # Z Translation in meters
    lowerBoundZ = mapmin[2]
    higherAngle_rx = mapmax[3]  # Rotation around X
    lowerAngle_rx = mapmin[3]
    higherAngle_ry = mapmax[4]  # Rotation around Y
    lowerAngle_ry = mapmin[4]
    higherAngle_rz = mapmax[5]  # Rotation around Z
    lowerAngle_rz = mapmin[5]

    all_best_solutions = [] # Store best solution every iteration

    # Problem Definition

    nVar = D            # Number of Decision Variables
    VarSize = [1, nVar]   # Size of Decision Variables Matrix
    stringcondition = "Max iterations reached"

    # PSO Parameters
    nPop = NPini
    minIt = 50  # Minimum number of iterations

    # velocity Limits. Definen cuanto se puede desplazar cada particula entre
    # iteraciones (en este caso un porcentaje 10% de la medida de cada dimension)

    VelMax_x = 0.1 * (higherBoundX - lowerBoundX)
    VelMin_x = -VelMax_x

    VelMax_y = 0.1 * (higherBoundY - lowerBoundY)
    VelMin_y = -VelMax_y

    VelMax_z = 0.1 * (higherBoundZ - lowerBoundZ)
    VelMin_z = -VelMax_z

    VelMax_Rx = 0.1 * (higherAngle_rx - lowerAngle_rx)
    VelMin_Rx = -VelMax_Rx

    VelMax_Ry = 0.1 * (higherAngle_ry - lowerAngle_ry)
    VelMin_Ry = -VelMax_Ry

    VelMax_Rz = 0.1 * (higherAngle_rz - lowerAngle_rz)
    VelMin_Rz = -VelMax_Rz

    # Initialization

    # Initialize Population
    swarm = Swarm(nPop, nVar)

    rndParticle = np.zeros(6)
    count = 0

    bestParticlecost = float('inf')
    worstParticlecost = float('inf')
    count_bestfix = 0
    count_worstfix = 0
    count_avgfix = 0
    ind_reparto_error = float('inf')
    particles = swarm.particles

    # Initialize position
    for current_iteration in range(nPop):
        if current_iteration == 0:  # first population is zero
            particles[current_iteration].position = np.zeros(6)
        else:
            for n in range(nVar):
                if n == 0:  # Translation
                    rndParticle[n] = np.random.uniform(lowerBoundX, higherBoundX)
                elif n == 1:
                    rndParticle[n] = np.random.uniform(lowerBoundY, higherBoundY)
                elif n == 2:
                    rndParticle[n] = np.random.uniform(lowerBoundZ, higherBoundZ)
                elif n == 3:  # Angle
                    rndParticle[n] = np.random.uniform(lowerAngle_rx, higherAngle_rx)
                elif n == 4:
                    rndParticle[n] = np.random.uniform(lowerAngle_ry, higherAngle_ry)
                elif n == 5:
                    rndParticle[n] = np.random.uniform(lowerAngle_rz, higherAngle_rz)

            particles[current_iteration].position = rndParticle.copy()

        # Initialize velocity to zero
        particles[current_iteration].velocity = np.zeros(VarSize)[0]

        # Evaluation poblacion inicial
        cand_scan = o3d.geometry.PointCloud()
        cand_scan.points = o3d.utility.Vector3dVector(spatial_rotation(scanCloud.points, particles[current_iteration].position))

        # Cortamos el mapa global a los limites de la nube local, para comparar con menos puntos(Puede salir nube vacía,
        # en tal caso crear nube de ceros)
        aabb = cand_scan.get_axis_aligned_bounding_box()
        cand_scan_min_bound = aabb.get_min_bound()
        cand_scan_max_bound = aabb.get_max_bound()

        cotout_region = o3d.geometry.AxisAlignedBoundingBox(min_bound=(cand_scan_min_bound), max_bound=(cand_scan_max_bound))
        Mapa_3D_cut = mapCloud.crop(cotout_region)

        if Mapa_3D_cut.is_empty():
            Mapa_3D_cut = o3d.geometry.PointCloud()
            points_array = np.zeros((np.array(cand_scan.points).shape[0], 3))
            Mapa_3D_cut.points = o3d.utility.Vector3dVector(points_array)

        # Busqueda de NN para cada punto del scan colocado en la localización candidata
        kdtree = o3d.geometry.KDTreeFlann(Mapa_3D_cut)
        Idx = np.empty((1, len(cand_scan.points)))

        for i in range(len(cand_scan.points)):
            query = np.array(cand_scan.points[i])
            query = np.where(np.isnan(query), 0, query) # Replace NaN for 0
            # Realizar la búsqueda de los vecinos más cercanos
            knn_sol = kdtree.search_knn_vector_3d(query, 1)
            index = knn_sol[1][0]
            Idx[0][i] = index

        # Crear matriz de correspondencia
        points_array = np.asarray(Mapa_3D_cut.points)

        # Ensure 'Idx' contains integer values
        Idx = Idx.astype(int)

        # Create matrix of correspondence
        correspondence_mat = np.zeros((Idx.shape[1], 3))
        for j in range(Idx.shape[1]):
            correspondence_mat[j, :] = points_array[Idx[0, j], :]

        # Calcular distancias euclídeas de cada punto del mapa y del scan al punto candidato
        dist_NNmap = distance_pc_to_point(correspondence_mat, particles[current_iteration].position)
        dist_scancand = distance_pc_to_point(cand_scan.points, particles[current_iteration].position)

        # Evaluar y asignar el error de las medidas (distancia euclídea o absoluta)
        particles[current_iteration].cost = costfunction3d(dist_scancand, dist_NNmap, version_fitness, err_dis)

        # Update Personal Best
        if particles[current_iteration].cost < particles[current_iteration].best_cost:
            particles[current_iteration].best_position = particles[current_iteration].position
            particles[current_iteration].best_cost = particles[current_iteration].cost

        # Update Global Best
        if particles[current_iteration].best_cost < swarm.global_best_cost:
            swarm.global_best_cost = particles[current_iteration].best_cost
            swarm.global_best_position = particles[current_iteration].best_position

        # Matriz para almacenar el mejor coste de cada iteración
        Bestcosts = np.zeros(iter_max)


    #.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#
    # Bucle principal del algoritmo PSO #
    #.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#
        
    for it in range(iter_max):
        start_time = time.time()
        for pop_id in range(nPop):

            # La nueva velocidad depende de su velocidad anterior, de su distancia a la mejor posición histórica de todo el enjambre y a la mejor
            # posición histórica de ella misma. Los coeficientes w, c1 y c2 son parámetros que cuantifican la importancia
            # que se le da a cada parte. np.random.rand() genera un valor entre 0-1 (para cada parámetro) que introduce una componente aleatoria.

            particles[pop_id].velocity = w * particles[pop_id].velocity \
                + c1 * np.random.rand(*VarSize) * (particles[pop_id].best_position- particles[pop_id].position) \
                + c2 * np.random.rand(*VarSize) * (swarm.global_best_position - particles[pop_id].position)


            particles[pop_id].velocity = particles[pop_id].velocity[0]

            # Apply velocity Limits

            particles[pop_id].velocity[0] = max(particles[pop_id].velocity[0], VelMin_x)
            particles[pop_id].velocity[0] = min(particles[pop_id].velocity[0], VelMax_x)

            particles[pop_id].velocity[1] = max(particles[pop_id].velocity[1], VelMin_y)
            particles[pop_id].velocity[1] = min(particles[pop_id].velocity[1], VelMax_y)

            particles[pop_id].velocity[2] = max(particles[pop_id].velocity[2], VelMin_z)
            particles[pop_id].velocity[2] = min(particles[pop_id].velocity[2], VelMax_z)

            particles[pop_id].velocity[3] = max(particles[pop_id].velocity[3], VelMin_Rx)
            particles[pop_id].velocity[3] = min(particles[pop_id].velocity[3], VelMax_Rx)

            particles[pop_id].velocity[4] = max(particles[pop_id].velocity[4], VelMin_Ry)
            particles[pop_id].velocity[4] = min(particles[pop_id].velocity[4], VelMax_Ry)

            particles[pop_id].velocity[5] = max(particles[pop_id].velocity[5], VelMin_Rz)
            particles[pop_id].velocity[5] = min(particles[pop_id].velocity[5], VelMax_Rz)

            # Update position
            particles[pop_id].position = np.array(particles[pop_id].position) + np.array(particles[pop_id].velocity)

            # velocity Mirror Effect
            IsOutside = (
                (particles[pop_id].position[0] < lowerBoundX) | (particles[pop_id].position[0] > higherBoundX) |
                (particles[pop_id].position[1] < lowerBoundY) | (particles[pop_id].position[1] > higherBoundY) |
                (particles[pop_id].position[2] < lowerBoundZ) | (particles[pop_id].position[2] > higherBoundZ) |
                (particles[pop_id].position[3] < lowerAngle_rx) | (particles[pop_id].position[3] > higherAngle_rx) |
                (particles[pop_id].position[4] < lowerAngle_ry) | (particles[pop_id].position[4] > higherAngle_ry) |
                (particles[pop_id].position[5] < lowerAngle_rz) | (particles[pop_id].position[5] > higherAngle_rz)
            )

            # Update velocity based on IsOutside
            if IsOutside:
                for j in range(6):
                    particles[pop_id].velocity[j] = -particles[pop_id].velocity[j]


            # Apply position Limits
            particles[pop_id].position[0] = max(min(particles[pop_id].position[0], higherBoundX), lowerBoundX)
            particles[pop_id].position[1] = max(min(particles[pop_id].position[1], higherBoundY), lowerBoundY)
            particles[pop_id].position[2] = max(min(particles[pop_id].position[2], higherBoundZ), lowerBoundZ)
            particles[pop_id].position[3] = max(min(particles[pop_id].position[3], higherAngle_rx), lowerAngle_rx)
            particles[pop_id].position[4] = max(min(particles[pop_id].position[4], higherAngle_ry), lowerAngle_ry)
            particles[pop_id].position[5] = max(min(particles[pop_id].position[5], higherAngle_rz), lowerAngle_rz)


            # Evaluación nuevamente
            cand_scan = o3d.geometry.PointCloud()
            cand_scan.points = o3d.utility.Vector3dVector(spatial_rotation(scanCloud.points, particles[pop_id].position))

            # Cortamos el mapa a los límites de la nube (puede salir nube vacía, poner a 000)
            aabb = cand_scan.get_axis_aligned_bounding_box()
            cand_scan_min_bound = aabb.get_min_bound()
            cand_scan_max_bound = aabb.get_max_bound()

            cotout_region = o3d.geometry.AxisAlignedBoundingBox(min_bound=(cand_scan_min_bound), max_bound=(cand_scan_max_bound))
            Mapa_3D_cut = mapCloud.crop(cotout_region)

            if Mapa_3D_cut.is_empty():
                Mapa_3D_cut = o3d.geometry.PointCloud()
                points_array = np.zeros((np.array(cand_scan.points).shape[0], 3))
                Mapa_3D_cut.points = o3d.utility.Vector3dVector(points_array)

            kdtree = o3d.geometry.KDTreeFlann(Mapa_3D_cut)
            Idx = np.empty((1, len(cand_scan.points)))

            for i in range(len(cand_scan.points)):
                query = np.array(cand_scan.points[i])
                query = np.where(np.isnan(query), 0, query) # Replace NaN for 0
                # Realizar la búsqueda de los vecinos más cercanos
                knn_sol = kdtree.search_knn_vector_3d(query, 1)
                index = knn_sol[1][0]
                Idx[0][i] = index

            # Crear matriz de correspondencia
            points_array = np.asarray(Mapa_3D_cut.points)

            # Ensure 'Idx' contains integer values
            Idx = Idx.astype(int)

            # Create matrix of correspondence
            correspondence_mat = np.zeros((Idx.shape[1], 3))
            for j in range(Idx.shape[1]):
                correspondence_mat[j, :] = points_array[Idx[0, j], :]

            # Calcular distancias euclídeas de cada punto del mapa y del scan al punto candidato
            dist_NNmap = distance_pc_to_point(correspondence_mat, particles[pop_id].position)
            dist_scancand = distance_pc_to_point(cand_scan.points, particles[pop_id].position)

            #print(f"dist_NNmap: {dist_NNmap}  |  dist_scancand: {dist_scancand}")

            particles[pop_id].cost = costfunction3d(dist_scancand, dist_NNmap, version_fitness, err_dis)

            # Update Personal Best
            if particles[pop_id].cost < particles[pop_id].best_cost:
                particles[pop_id].best_position = particles[pop_id].position
                particles[pop_id].best_cost = particles[pop_id].cost

            # Update Global Best
            if particles[pop_id].best_cost < swarm.global_best_cost:
                swarm.global_best_cost = particles[pop_id].best_cost
                swarm.global_best_position = particles[pop_id].best_position

            #print(f"It: {it} | particle[{pop_id}].cost: " + str(particle[pop_id].cost) + " GlobalBest.cost: " + str(GlobalBest.cost))


        Bestcosts[it] = swarm.global_best_cost
        # Reducir la inercia
        w *= wdamp

        # Analizar la población
        sumcosts = 0  # costo promedio
        for j in range(nPop):
            sumcosts += particles[j].cost

        id = 0
        worst = 0
        worst_id = 0
        for p in particles:
            if p.cost > worst:
                worst = p.cost
                worst_id = id
            id += 1

        bestParticlecostnow = min(p.cost for p in particles)
        worstParticlecostnow = max(p.cost for p in particles)

        # Display evolution each 10 iterations
        if count == 10:
            print(f"\nIt: {it}, {Color.GREEN}Best: {round(swarm.global_best_cost, 4)}{Color.END}, {Color.RED}Worst: {round(worstParticlecost,4)}{Color.END}, {Color.YELLOW}Average: {round(sumcosts/nPop,4)}{Color.END}, Best/measure: {round(bestParticlecost/nPop,4)}, Worst/best: {round(worstParticlecost/bestParticlecost,4)}, Avg/best: {round(sumcosts/nPop/bestParticlecost,4)} \n position (x, y, z, alpha, beta, theta): [{round(swarm.global_best_position[0],4)}, {round(swarm.global_best_position[1],4)}, {round(swarm.global_best_position[2],4)}, {round(swarm.global_best_position[3],4)}, {round(swarm.global_best_position[4],4)}, {round(swarm.global_best_position[5],4)}]\n")
            count=0
        count=count+1
        end_time = time.time()
        
        #print(f'Count: {count} in {round(end_time-start_time, 2)} seconds') # DEBUG

        # Convergence Indicators
        if bestParticlecostnow < bestParticlecost:  # ¿Mejora el mejor respecto a la iteración anterior?
            count_worstfix = 0
            count_avgfix = 0
            count_bestfix = 0  # Sí, reiniciar contador a 0
        else:
            count_bestfix += 1  # No, incrementar contador de veces que no mejora

        bestParticlecost = bestParticlecostnow


        if worstParticlecost > worstParticlecostnow:  # ¿Mejora el peor candidato?
            count_worstfix = 0
            count_avgfix = 0
            count_bestfix = 0  # Sí, reiniciar contador a 0
        else:
            count_worstfix += 1  # No, incrementar contador de veces que no mejora

        worstParticlecost = worstParticlecostnow

        ind_reparto_error_aux = sumcosts / (nPop * bestParticlecost)

        if ind_reparto_error_aux < ind_reparto_error:  # ¿Mejora la media?
            count_avgfix = 0
            count_worstfix = 0
            count_bestfix = 0  # Sí, reiniciar contadores
        else:
            count_avgfix += 1  # No, incrementar contador de veces que no mejora

        ind_reparto_error = ind_reparto_error_aux

        #print(f"It: {it}, Time: {round(end_time-start_time,2)}  |||  count_bestfix: {count_bestfix}, count_worstfix: {count_worstfix}, count_avgfix: {count_avgfix}  |||  worstParticlecost: {worstParticlecost} | bestParticlecost: {bestParticlecost} | ind_reparto_error: {ind_reparto_error}")

        if (all([p.cost for p in particles]) == swarm.global_best_cost) or \
        ((count_bestfix > 10 and count_worstfix > 10 and count_avgfix > 10) and it >= minIt) or \
        ((worstParticlecost / bestParticlecost < 1.15 and ind_reparto_error < 1.15) and it >= minIt):

            if all([p.cost for p in particles]) == swarm.global_best_cost:
                stringcondition = 'total convergence'
            elif worstParticlecost / bestParticlecost < 1.15 and ind_reparto_error < 1.15:
                stringcondition = 'normal convergence'
            elif count_bestfix > 10 and count_worstfix > 10 and count_avgfix > 10:
                stringcondition = 'invariant convergence'

            print(f'Population converged in: {it} iterations and condition: {stringcondition}')
            break

    
        # Save current best solution
        all_best_solutions.append(swarm.global_best_position)
    ########################################################
    ########################################################
    
    # BestParticle = swarm.global_best_position
    bestcost = swarm.global_best_cost
    rmse_array =  Bestcosts

    pcAligned = o3d.geometry.PointCloud()
    pcAligned.points = o3d.utility.Vector3dVector(spatial_rotation(scanCloud.points, all_best_solutions[-1]))

    return(pcAligned, all_best_solutions, bestcost, rmse_array, it, stringcondition)