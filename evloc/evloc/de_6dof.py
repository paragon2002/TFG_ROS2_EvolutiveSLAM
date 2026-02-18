import numpy as np
import open3d as o3d
import time
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
        # Crea una nueva instancia de la clase Population con los mismos valores
        return Population(self.Position.copy(), self.Cost.copy())

def de_6dof(scanCloud,mapCloud,mapmax,mapmin,err_dis,NPini,D,iter_max,F,CR,version_fitness):
    """
    Differential Evolution with Thresholding and Discarding 
    Evolucion por mutación. En cada iteración, cada candidato (xi) genera uno
    nuevo (x(i+1)). Este nuevo es una combinación parámetro a parámetro del candidato
    antiguo y de una combinación tal que x(i+1)=F*(xc-xb)+xa, de otros 3 candidados de la poblacion xa,
    xb y xc escogidos aleatoriamente.
    El factor de mutación F determina como de "lejos" puede terminar cada nuevo parámetro en caso de mutar
    La tasa de cruce CR define qué porcentaje de parámetros de x(i+1) son mutados o se heredan
    de xi.
    """
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
    minIt = 50  # Minimum number of iterations
    stringcondition = "Max iterations reached"
    # DE Parameters
    # F - Mutation
    # CR - Crossover rate

    # Initialization
    # Empty Candidate Structure
    empty_population = Population()

    # Initialize Population
    population = [empty_population.clone() for _ in range(NPini)]
    rndmember = np.zeros(6)
    count = 0
    vis1 = 1
    vis2 = 1
    best_particle_cost = 100000000
    worst_particle_cost = 100000
    count_bestfix = 0  # Counters for algorithm convergence
    count_worstfix = 0
    count_avgfix = 0
    ind_reparto_error = 100000
    NP = NPini

    ########## LOOP 1 ###########
    for current_iteration in range(NPini):
        # Initialize Position
        if current_iteration == 0:  # first population is a vector of zeros
            population[current_iteration].Position = np.zeros(6)
        else:
            rndmember = np.zeros(6)
            for n in range(nVar):
                if n == 0:  # Translation
                    rndmember[n] = np.random.uniform(lowerBoundX, higherBoundX)
                elif n == 1:
                    rndmember[n] = np.random.uniform(lowerBoundY, higherBoundY)
                elif n == 2:
                    rndmember[n] = np.random.uniform(lowerBoundZ, higherBoundZ)
                elif n == 3:  # Angle
                    rndmember[n] = np.random.uniform(lowerAngle_rx, higherAngle_rx)
                elif n == 4:
                    rndmember[n] = np.random.uniform(lowerAngle_ry, higherAngle_ry)
                elif n == 5:
                    rndmember[n] = np.random.uniform(lowerAngle_rz, higherAngle_rz)

            population[current_iteration].Position = rndmember

        # Transform local cloud into each candidate's location
        cand_scan = o3d.geometry.PointCloud()
        cand_scan.points = o3d.utility.Vector3dVector(spatial_rotation(scanCloud.points, population[current_iteration].Position))

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
        dist_NNmap = distance_pc_to_point(correspondence_mat, population[current_iteration].Position)
        dist_scancand = distance_pc_to_point(cand_scan.points, population[current_iteration].Position)
        
        # Evaluar y asignar el error de las medidas (distancia euclídea o absoluta)
        population[current_iteration].Cost = costfunction3d(dist_scancand, dist_NNmap, version_fitness, err_dis)

    ###### END LOOP 1

    #.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#
    # Bucle principal del algoritmo DE  #
    #.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#
    for it in range(iter_max):
        start_time = time.time()
        for pop_id in range(NP-1):
            # Mutación y cruce
            a, b, c = np.random.randint(0, NP, size=3)
            newmember = Population()
            newmember.Position = [0,0,0,0,0,0]

            for j in range(nVar): # that mutation only takes place a CR% of times, if not, that parameter remains from the original candidate xi
                if np.random.rand() < CR:
                    newmember.Position[j] = population[c].Position[j] + F * (population[a].Position[j] - population[b].Position[j])
                else:
                    newmember.Position[j] = population[pop_id].Position[j]

            # Apply limits for new member
            newmember.Position[0] = max(newmember.Position[0], lowerBoundX)
            newmember.Position[0] = min(newmember.Position[0], higherBoundX)

            newmember.Position[1] = max(newmember.Position[1], lowerBoundY)
            newmember.Position[1] = min(newmember.Position[1], higherBoundY)

            newmember.Position[2] = max(newmember.Position[2], lowerBoundZ)
            newmember.Position[2] = min(newmember.Position[2], higherBoundZ)

            newmember.Position[3] = max(newmember.Position[3], lowerAngle_rx)
            newmember.Position[3] = min(newmember.Position[3], higherAngle_rx)

            newmember.Position[4] = max(newmember.Position[4], lowerAngle_ry)
            newmember.Position[4] = min(newmember.Position[4], higherAngle_ry)

            newmember.Position[5] = max(newmember.Position[5], lowerAngle_rz)
            newmember.Position[5] = min(newmember.Position[5], higherAngle_rz)

            # Evaluación nuevamente
            cand_scan = o3d.geometry.PointCloud()
            cand_scan.points = o3d.utility.Vector3dVector(spatial_rotation(scanCloud.points, newmember.Position))

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
            dist_NNmap = distance_pc_to_point(correspondence_mat, newmember.Position)
            dist_scancand = distance_pc_to_point(cand_scan.points, newmember.Position)

            # Evaluar y asignar el error de las medidas (distancia euclídea o absoluta)
            newmember.Cost = costfunction3d(dist_scancand, dist_NNmap, version_fitness, err_dis)

            # Actualizar el miembro si mejora
            if newmember.Cost < population[pop_id].Cost * 0.98:  # con umbral, el nuevo miembro debe mejorar más del 2% (evita efecto de ruido)
                population[pop_id] = newmember

        # Discarding, substituting the worst candidates for some of the best
        # members (speeds up convergence)

        disc_range=0.9 # percentage to keep, in this case first 90%, last 10% will be replaced
        repl_range=0.2 # percentage to repalce with, in this case first 20%

        population.sort(key=lambda x: x.Cost)

        # Discard a portion of the population
        for i in range(NP, int(disc_range * NP), -1):
            population[i-1] = population[np.random.randint(0, repl_range*NP)]

        # Sort the population based on 'Cost'
        population = sorted(population, key=lambda obj: obj.Cost)

        BestSol  = population[0]
        WorstSol = population[NP-1]
        sumcosts=0

        # Calculate average cost
        for k in range(NP):
            pop = population[k]
            sumcosts += pop.Cost
        average_cost = sumcosts / NP

        # Best and worst costs
        best_particle_cost_now = BestSol.Cost
        worst_particle_cost_now = WorstSol.Cost

        if count == 10:
            print(f"\nIt: {it}, {Color.GREEN}Best: {round(best_particle_cost_now, 4)}{Color.END}, {Color.RED}Worst: {round(worst_particle_cost_now,4)}{Color.END}, {Color.YELLOW}Average: {round(average_cost,4)}{Color.END}, Best/measure: {round(best_particle_cost_now/NP,4)}, Worst/best: {round(worst_particle_cost_now/best_particle_cost_now,4)}, Avg/best: {round(average_cost/NP/best_particle_cost_now,4)} \n Position (x, y, z, alpha, beta, theta): [{round(BestSol.Position[0],4)}, {round(BestSol.Position[1],4)}, {round(BestSol.Position[2],4)}, {round(BestSol.Position[3],4)}, {round(BestSol.Position[4],4)}, {round(BestSol.Position[5],4)}]\n")
            count=0
        count=count+1
        end_time = time.time()
        
        #print(f'Count: {count} in {round(end_time-start_time, 2)} seconds') # DEBUG

        # Convergence indicators
        if best_particle_cost_now < best_particle_cost:
            count_worstfix = 0
            count_avgfix = 0
            count_bestfix = 0  # Yes, improvement from the previous iteration
        else:
            count_bestfix += 1  # No, increment the counter for non-improvement

        best_particle_cost = best_particle_cost_now

        # Check if the worst candidate has improved
        if worst_particle_cost_now > worst_particle_cost:
            count_worstfix = 0
            count_avgfix = 0
            count_bestfix = 0  # Yes, improvement in the worst candidate
        else:
            count_worstfix += 1  # No, increment the counter for non-improvement

        worst_particle_cost = worst_particle_cost_now
 
        ind_reparto_error_aux = sumcosts / (NP*best_particle_cost)

        if (ind_reparto_error_aux < ind_reparto_error): #Mejora la media?
            count_avgfix=0
            count_worstfix=0
            count_bestfix=0  #si
        else:
            count_avgfix= count_avgfix+1 #no

        ind_reparto_error = ind_reparto_error_aux

        # Modificacion de parámetros del algoritmo en caliente
        if (worst_particle_cost/best_particle_cost < 2.5) and (ind_reparto_error < 2): #Reducción del factor de mutación(lo lejos que se mueve un candidato) si converge un poco
            F = 0.7
            if (vis1 == 1):
                print(Color.CYAN + f'F reduced to 0.7' + Color.END)
                vis1 = 0
        if (worst_particle_cost/best_particle_cost < 1.5) and (ind_reparto_error < 1.25): # Reducción mayor si converge mucho, busqueda más cerca de las pos. actuales
            F = 0.3
            NP = int(NPini/5)
            if (vis2 == 1):
               print(Color.CYAN + f'F reduced to 0.3' + Color.END)
               vis2 = 0

        # Condiciones de convergencia (todos costes iguales||poblacion mejor,
        # media y peor muy parecida || poblacion estancada || máximo de iteraciones)
        if all(obj.Cost == best_particle_cost_now for obj in population) or \
                (worst_particle_cost / best_particle_cost < 1.15 and ind_reparto_error < 1.15 and it >= minIt) or \
                (count_bestfix > 10 and count_worstfix > 10 and count_avgfix > 10 and it >= minIt):
            
            if all(obj.Cost == best_particle_cost for obj in population):
                stringcondition = 'total convergence'
            elif worst_particle_cost / best_particle_cost < (1.15 + err_dis) and ind_reparto_error < (1.15 + err_dis):
                stringcondition = 'normal convergence'
            elif count_bestfix > 10 and count_worstfix > 10 and count_avgfix > 10:
                stringcondition = 'invariant convergence'
            
            print(f'\n{Color.CYAN}Population converged in: {it} iterations and condition: {stringcondition}{Color.END}')
            break


        # Save current best solution
        all_best_solutions.append(BestSol.Position)
    ########################################################
    ########################################################
        
    # BestMember = BestSol.Position

    rmse_array =  BestSol.Cost

    bestCost = BestSol.Cost

    pcAligned = o3d.geometry.PointCloud()
    pcAligned.points = o3d.utility.Vector3dVector(spatial_rotation(scanCloud.points, all_best_solutions[-1]))

    return(pcAligned, all_best_solutions, bestCost, rmse_array, it, stringcondition)