import numpy as np
import time
from math import pi
from evloc.common_classes import Color
from evloc.common_classes import Solution
from evloc.common_functions import add_noise_to_pc
from evloc.de_6dof import de_6dof
from evloc.pso_6dof import pso_6dof
from evloc.iwo_6dof import iwo_6dof

def gl_6dof(map_global, scancloud, groundtruth, algorithm, version_fitness, err_dis,unif_noise):
    """
    Global Localization Algorithm based on evolutonary metaheuristics

    Definimos los límites de búsqueda para cada grado de libertad, limites del
    mapa en traslación +- 6 grados para pitch y roll y 360º para yaw

    Get the axis-aligned bounding box
    """

    aabb = map_global.get_axis_aligned_bounding_box()

    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()

    x_min, y_min, z_min = min_bound
    x_max, y_max, z_max = max_bound

    mapmin=[x_min, y_min, z_min, -0.1, -0.1,-pi]
    mapmax=[x_max, y_max, z_max, 0.1, 0.1, pi]

    real_scan = add_noise_to_pc(scancloud, err_dis, unif_noise) # Add enviroment and sensor noises to scan


    print(Color.GREEN + f'\nPosicion real del robot [x, y, z, alpha, beta, theta]: ' +
        f'[{round(groundtruth[0], 4)}, {round(groundtruth[1], 4)}, {round(groundtruth[2], 4)}, ' +
        f'{round(groundtruth[3], 4)}, {round(groundtruth[4], 4)}, {round(groundtruth[5], 4)}]' + Color.END)
    
    initial_time = time.time()

    #--------------------------------------------------------------------------------------------
    # EXECUTION OF THE EVOLUTIVE ALGORITHM

    if (algorithm.type == 1): # Differential Evolution
        NPini=algorithm.NPini
        iter_max=algorithm.iter_max
        D=algorithm.D
        F=algorithm.F
        CR=algorithm.CR
        [pcAligned, allEstimates, bestCost, rmse_array, it, stop_condition] = de_6dof(real_scan, map_global,mapmax,mapmin,err_dis,NPini,D,iter_max,F,CR,version_fitness)

    elif (algorithm.type == 2): # PSO
        NPini=algorithm.NPini
        iter_max=algorithm.iter_max
        D=algorithm.D
        w=algorithm.w
        wdamp=algorithm.wdamp
        c1=algorithm.c1
        c2=algorithm.c2

        [pcAligned, allEstimates, bestCost, rmse_array, it, stop_condition] = pso_6dof(real_scan,map_global,mapmax, mapmin, err_dis, NPini, D ,w, wdamp, c1, c2, iter_max,version_fitness)

    elif (algorithm.type == 3): # IWO
        NPini=algorithm.NPini
        iter_max=algorithm.iter_max
        D=algorithm.D
        Smin=algorithm.Smin
        Smax=algorithm.Smax
        exponent=algorithm.exponent
        sigma_initial=algorithm.sigma_initial
        sigma_final=algorithm.sigma_final
        [pcAligned, allEstimates, bestCost, rmse_array, it, stop_condition] = iwo_6dof(real_scan, map_global, mapmax, mapmin, err_dis, NPini, D , Smin, Smax, exponent, sigma_initial, sigma_final, iter_max, version_fitness)

    estimate = allEstimates[-1]

    final_time = time.time()
    
    # Display results
    print(f'\nPosicion real del robot[x, y, z, alpha, beta, theta]: ' +
        f'[{round(groundtruth[0], 4)}, {round(groundtruth[1], 4)}, {round(groundtruth[2], 4)}, ' +
        f'{round(groundtruth[3], 4)}, {round(groundtruth[4], 4)}, {round(groundtruth[5], 4)}]')
    
    print(f'\nPosicion estimada tras ejecución: ' +
        f'[{round(estimate[0], 4)}, {round(estimate[1], 4)}, {round(estimate[2], 4)}, ' +
        f'{round(estimate[3], 4)}, {round(estimate[4], 4)}, {round(estimate[5], 4)}]')

    all_poserrors = [groundtruth[0] - estimate[0],
                     groundtruth[1] - estimate[1],
                     groundtruth[2] - estimate[2]
    ]
    
    all_orierrors = [
        abs((groundtruth[3] - estimate[3]) * 180 / pi),
        abs((groundtruth[4] - estimate[4]) * 180 / pi),
        abs((groundtruth[5] - estimate[5]) * 180 / pi)
    ]

    print(Color.PURPLE + f'\nEl error de posicion es: [{round(all_poserrors[0],4)}, {round(all_poserrors[1],4)}, {round(all_poserrors[2],4)}] m, y el de orientacion: [{round(all_orierrors[0],4)}, {round(all_orierrors[1],4)}, {round(all_orierrors[2],4)}] grados' + Color.END)
    print(Color.GREEN + f'Tiempo transcurrido: {round(final_time-initial_time, 2)} segundos' + Color.END)
    solution = Solution(it, (final_time-initial_time), allEstimates, all_poserrors, all_orierrors, map_global, real_scan.points, stop_condition)

    return solution