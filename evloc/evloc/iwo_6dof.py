import numpy as np
import open3d as o3d
import time
from evloc.common_functions import costfunction3d
from evloc.common_functions import distance_pc_to_point
from evloc.common_functions import spatial_rotation
from evloc.common_classes import Color

class Plant:
    def __init__(self, position=[], cost=[]):
        self.Position = position
        self.Cost = cost
        
    def clone(self):
        # Crea una nueva instancia de la clase Population con los mismos valores
        return Plant(self.Position.copy(), self.Cost.copy())

def iwo_6dof(scanCloud, mapCloud, mapmax, mapmin, err_dis, NPini, D , Smin, Smax, exponent, sigma_initial, sigma_final, iter_max, version_fitness):
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

    nVar = D  # Number of Decision Variables
    VarSize = [1, nVar]  # Decision Variables Matrix Size
    minIt = 50
    stringcondition = "Max iterations reached"

    # IWO Parameters

    nPop = NPini;      # Maximum Population Size
    nPop0=int(nPop/2.5);    # Initial population size

    empty_plant = Plant()

    # Initialize Population
    pop = [empty_plant.clone() for _ in range(nPop0)]  # Initial Population Array
    rndMember = np.zeros(D)
    count = 0  # contadores a cero y costes a infinito
    bestParticleCost = 100000000
    worstParticleCost = 100000
    count_bestfix = 0  # contadores para la convergencia del algoritmo
    count_worstfix = 0
    count_avgfix = 0
    ind_reparto_error = 100000

    ########## LOOP 1 ###########
    # Initialize Position
    for current_iteration in range(len(pop)):
        if current_iteration == 0:  # first population is zero
            pop[current_iteration].Position = np.zeros(nVar)
        else:
            for n in range(nVar):
                if n == 0:  # Translation
                    rndMember[n] = np.random.uniform(lowerBoundX, higherBoundX)
                elif n == 1:
                    rndMember[n] = np.random.uniform(lowerBoundY, higherBoundY)
                elif n == 2:
                    rndMember[n] = np.random.uniform(lowerBoundZ, higherBoundZ)
                elif n == 3:  # Angle
                    rndMember[n] = np.random.uniform(lowerAngle_rx, higherAngle_rx)
                elif n == 4:
                    rndMember[n] = np.random.uniform(lowerAngle_ry, higherAngle_ry)
                elif n == 5:
                    rndMember[n] = np.random.uniform(lowerAngle_rz, higherAngle_rz)
            pop[current_iteration].Position = rndMember

        # Transform local cloud into each candidate's location
        cand_scan = o3d.geometry.PointCloud()
        cand_scan.points = o3d.utility.Vector3dVector(spatial_rotation(scanCloud.points, pop[current_iteration].Position))

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
        dist_NNmap = distance_pc_to_point(correspondence_mat, pop[current_iteration].Position)
        dist_scancand = distance_pc_to_point(cand_scan.points, pop[current_iteration].Position)
        
        # Evaluar y asignar el error de las medidas (distancia euclídea o absoluta)
        pop[current_iteration].Cost = costfunction3d(dist_scancand, dist_NNmap, version_fitness, err_dis)

        # Matriz para almacenar el mejor coste de cada iteración
        BestCosts = np.zeros((iter_max, 1))

    for it in range(iter_max):
        # Update Standard Deviation
        # La distancia a la que se envían las semillas es una campana de Gauss
        # entorno a la posición de la planta. Esta campana va disminuyendo cada
        # iteración, cada vez las semillas caen más cerca de la planta (busqueda
        # más local).
        # La desviación estandar de la campana de gauss decrece dependiendo de
        # la iteracion en la que estemos respecto a las máximas, entre una
        # desviación máxima y una mínima
        sigma = ((iter_max - it) / (iter_max - 1))**exponent * (sigma_initial - sigma_final) + sigma_final

        # Obtener los mejores y peores valores de costo
        Costs = [p.Cost for p in pop]
        BestCost = min(Costs)
        WorstCost = max(Costs)

        # Inicializar la población de descendencia
        newpop = []
    
        start_time = time.time()

        for pop_id in range(len(pop)):
            # Asignación de número de semillas para cada planta
            # Proporcional entre un máximo y un mínimo según la posición del
            # coste de la planta entre el mejor y el peor en ese momento. Regla
            # de 3.

            # Cálculo de ratio
            ratio = (pop[pop_id].Cost - WorstCost) / (BestCost - WorstCost)
            S = int(Smin + (Smax - Smin) * ratio)
            for j in range(1, S + 1):
                # Initialize Offspring
                newsol = empty_plant.clone()

                # Generate Random Location
                newsol.Position = (pop[pop_id].Position + sigma * np.random.randn(*VarSize))[0]

                # Apply Lower/Upper Bounds
                newsol.Position[0] = max(newsol.Position[0], lowerBoundX)
                newsol.Position[0] = min(newsol.Position[0], higherBoundX)

                newsol.Position[1] = max(newsol.Position[1], lowerBoundY)
                newsol.Position[1] = min(newsol.Position[1], higherBoundY)

                newsol.Position[2] = max(newsol.Position[2], lowerBoundZ)
                newsol.Position[2] = min(newsol.Position[2], higherBoundZ)

                newsol.Position[3] = max(newsol.Position[3], lowerAngle_rx)
                newsol.Position[3] = min(newsol.Position[3], higherAngle_rx)

                newsol.Position[4] = max(newsol.Position[4], lowerAngle_ry)
                newsol.Position[4] = min(newsol.Position[4], higherAngle_ry)

                newsol.Position[5] = max(newsol.Position[5], lowerAngle_rz)
                newsol.Position[5] = min(newsol.Position[5], higherAngle_rz)

                # Evaluate Offspring
                # Transform local cloud into each candidate's location
                cand_scan = o3d.geometry.PointCloud()
                cand_scan.points = o3d.utility.Vector3dVector(spatial_rotation(scanCloud.points, newsol.Position))

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
                dist_NNmap = distance_pc_to_point(correspondence_mat, newsol.Position)
                dist_scancand = distance_pc_to_point(cand_scan.points, newsol.Position)
                
                #evaluacion devuelve el error de las medidas para el punto perturbado
                newsol.Cost = costfunction3d(dist_scancand, dist_NNmap, version_fitness, err_dis)

                # Add Offspring to the Population
                newpop.append(newsol)

        # Merge Populations
        pop += newpop

        # Sort the population based on 'Cost'
        pop = sorted(pop, key=lambda obj: obj.Cost)

        # Competitive Exclusion (Delete Extra Members)
        if len(pop) > nPop:
            pop = pop[:nPop]

        # Store Best Solution Ever Found
        BestSol = pop[0]

        # Store Best Cost History
        BestCosts[it] = BestSol.Cost

        # Analyze population
        sumcosts = 0  # Average cost
        for j in range(len(pop)):
            sumcosts += pop[j].Cost

        # Best and worst
        bestParticleCostnow = min([p.Cost for p in pop])
        worstParticleCostnow = max([p.Cost for p in pop])

        # Display evolution each 10 iterations
        if count == 10:
            print(f"\nIt: {it}, {Color.GREEN}Best: {round(bestParticleCostnow, 4)}{Color.END}, {Color.RED}Worst: {round(worstParticleCostnow,4)}{Color.END}, {Color.YELLOW}Average: {round(sumcosts/nPop,4)}{Color.END}, Best/measure: {round(bestParticleCostnow/nPop,4)}, Worst/best: {round(worstParticleCostnow/bestParticleCostnow,4)}, Avg/best: {round(sumcosts/nPop/bestParticleCostnow,4)} \n Position (x, y, z, alpha, beta, theta): [{round(BestSol.Position[0],4)}, {round(BestSol.Position[1],4)}, {round(BestSol.Position[2],4)}, {round(BestSol.Position[3],4)}, {round(BestSol.Position[4],4)}, {round(BestSol.Position[5],4)}]\n")
            count = 0
        count += 1

        end_time = time.time()
        #print(f'Count: {count} in {round(end_time-start_time, 2)} seconds') # DEBUG

        # Convergence indicators
        if bestParticleCostnow < bestParticleCost:  # ¿Mejora el mejor respecto a la anterior iteración?
            count_worstfix = 0
            count_avgfix = 0
            count_bestfix = 0  # Si, contador a 0
        else:
            count_bestfix = count_bestfix + 1  # no, contador de veces que no mejora

        bestParticleCost=bestParticleCostnow

        if worstParticleCost > worstParticleCostnow:  # ¿mejora el peor candidato?
            count_worstfix = 0
            count_avgfix = 0
            count_bestfix = 0  # Sí, reiniciar contadores
        else:
            count_worstfix += 1  # No, incrementar contador de veces que no mejora

        worstParticleCost = worstParticleCostnow

        ind_reparto_error_aux = sumcosts / (nPop * bestParticleCost)

        if ind_reparto_error_aux < ind_reparto_error:  # ¿Mejora la media?
            count_avgfix = 0
            count_worstfix = 0
            count_bestfix = 0  # Sí, reiniciar contadores
        else:
            count_avgfix += 1  # No, incrementar contador

        ind_reparto_error = ind_reparto_error_aux

        # Condiciones de convergencia (todos los costes iguales, población mejor, media y peor muy parecidas, población estancada, máximo de iteraciones)
        if all([p.Cost for p in pop] == bestParticleCost) or ((worstParticleCost / bestParticleCost) < 1.15 and ind_reparto_error < 1.15) and it >= minIt or (count_bestfix > 10 and count_worstfix > 10 and count_avgfix > 10) and it >= minIt:
            if all([p.Cost for p in pop] == bestParticleCost):
                stringcondition = 'total convergence'
            elif (worstParticleCost / bestParticleCost) < (1.15 + err_dis) and ind_reparto_error < 1.15:
                stringcondition = 'normal convergence'
            elif count_bestfix > 10 and count_worstfix > 10 and count_avgfix > 10:
                stringcondition = 'invariant convergence'
                
            print(f'Population converged in: {it} iterations and condition: {stringcondition}')
            break

        # Save current best solution
        all_best_solutions.append(BestSol.Position)
    ########################################################
    ########################################################
        
    # BestWeed = BestSol.Position

    rmse_array =  BestCost

    bestCost = BestSol.Cost

    pcAligned = o3d.geometry.PointCloud()
    pcAligned.points = o3d.utility.Vector3dVector(spatial_rotation(scanCloud.points, all_best_solutions[-1]))

    return(pcAligned, all_best_solutions, bestCost, rmse_array, it, stringcondition)

######################################################################