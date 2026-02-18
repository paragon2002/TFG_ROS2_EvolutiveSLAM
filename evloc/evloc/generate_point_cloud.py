import numpy as np
from math import pi
import os
from evloc.common_classes import Color
from evloc.gl_6dof import gl_6dof
from evloc.save_data import save_error_data
from evloc.common_classes import Algorithm
from evloc.evloc_constants import *

def generate_point_cloud(auto=False,
                         id_cloud = 9,
                         err_dis = 0, 
                         unif_noise = 0,
                         algorithm_type = 1,
                         version_fitness = 1,
                         user_NPini = 100,
                         user_iter_max = 500,
                         D=6,
                         F=0.9,
                         CR=0.75,
                         w=1,
                         wdamp=0.99,
                         c1=1.5,
                         c2=2,
                         Smin=0,
                         Smax=4,
                         exponent=2,
                         sigma_initial=0.5,
                         sigma_final=0.001,
                         map_global=None,
                         real_scan=None,
                         groundtruth=None
                         ):
    """
    Executes the evolutive localization algorithm.
    Returns the points that form the calculated point cloud.
    """

    # Variables introduced via keyboard # (Only if not in auto mode)
    if (auto):
        print("\n" + Color.DARKCYAN + f"Auto mode enabled. Cloud {id_cloud}/{len(os.listdir(LOCAL_CLOUDS_FOLDER))}" + Color.END)

    print(Color.BOLD + "\nFINAL ALGORITHM PARAMETERS: " + Color.END)

    if algorithm_type == 1:
        print(f"Algortihm type: 1 (DE)")
    elif algorithm_type == 2:
        print(f"Algortihm type: 2 (PSO)")
    elif algorithm_type == 3:
        print(f"Algortihm type: 3 (IWO)")

    print(f"Local Cloud: {id_cloud}")
    print(f"Sensor Error: {err_dis}")
    print(f"Uniform Noise: {unif_noise}")
    print(f"NPini: {user_NPini}")
    print(f"iter_max: {user_iter_max}")
    print(f"D: {D}")

    if algorithm_type == 1:
        print(f"F: {F}")
        print(f"CR: {CR}")
        
    if algorithm_type == 2:
        print(f"w: {w}")
        print(f"wdamp: {wdamp}")
        print(f"c1: {c1}")
        print(f"c2: {c2}")

    if algorithm_type == 3:
        print(f"Smin: {Smin}")
        print(f"Smax: {Smax}")
        print(f"exponent: {exponent}")
        print(f"sigma_initial: {sigma_initial}")
        print(f"sigma_final: {sigma_final}")

    algorithm = Algorithm(type=algorithm_type, NPini=user_NPini, iter_max=user_iter_max, D=D,F=F, CR=CR, w=w, wdamp=wdamp, c1=c1, c2=c2,
                          Smin=Smin, Smax=Smax, exponent=exponent, sigma_initial=sigma_initial, sigma_final=sigma_final)

    solution = gl_6dof(map_global, real_scan, groundtruth, algorithm, version_fitness, err_dis, unif_noise)

    save_error_data(id_cloud, algorithm_type, user_NPini, user_iter_max, D, F, CR, solution.time, solution.it, solution.all_poserrors, solution.all_orierrors,
                    w, wdamp, c1, c2, Smin, Smax, exponent, sigma_initial, sigma_final, solution.stop_condition)

    return solution.all_pose_estimates