import rclpy 
from rclpy.node import Node
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
import numpy as np
import open3d as o3d
import csv
import os
import time
from math import pi

from evloc.common_classes import Color
from evloc.evloc_constants import *

def ask_params(local_clouds_folder=None,online=False):
    """
    Asks the user for the desired algorithm parameters and returns them.
    local cloud ID,
    laser error,
    uniform error,
    population size,
    max iterations,
    algortihm type and version fitness are set to default.
    """
    
    id_cloud = None
    num_clouds=None
    if not online:
        # Local Cloud
        num_clouds = len(os.listdir(local_clouds_folder))

        print(Color.BOLD + f'\nAvailable scans [1-{num_clouds}]' + Color.END)
        id_cloud = input(Color.BOLD + "Select cloud as real scan: " + Color.END)
        if not id_cloud.strip():
            id_cloud = 9
            print(f'Default selected cloud: {id_cloud}')
        try:
            if int(id_cloud) > num_clouds or int(id_cloud) < 1:
                print(f'Error. Selected cloud ({id_cloud}) does not exist.') 
                exit(1)

        except ValueError as e:
            print(f'Error: Invalid Number. {e}')
            exit(1)

            id_cloud = int(id_cloud)

    # Simulated laser error
    err_dis = input(Color.BOLD + "\nSensor noise (%): " + Color.END)
    if not err_dis.strip():
        err_dis = 0
        print(f'Default Noise: {err_dis}%')
    else:
        try:
            err_dis = int(err_dis)
            
            if err_dis > 100 or err_dis < 0:
                print(f'Error. Selected error ({err_dis}) is invalid.') 
                exit(1)

            err_dis = err_dis/100

        except ValueError as e:
            print(f'Error: Invalid Input. {e}')
            exit(1)

    # Simulated environmental noise
    unif_noise = input(Color.BOLD + "\nEnvironmental noise (Uniform distribution) (%): " + Color.END)
    if not unif_noise.strip():
        unif_noise = 0
        print(f'Default Noise: 0%')
    else:
        try:
            unif_noise = int(unif_noise)
            
            if unif_noise > 100 or unif_noise < 0:
                print(f'Error. Selected error ({unif_noise}) is invalid.') 
                exit(1)

            unif_noise = unif_noise/100

        except ValueError as e:
            print(f'Error: Invalid Input. {e}')
            exit(1)

    # Algorithm selection
    algorithm_type = input(Color.BOLD + "\nIntroduce the Evolutionary Algorithm that you want to apply: \n 1) DE - Differential Evolution (Default). \
                            \n 2) PSO - Particle Swarm Optimization.\n 3) IWO - Invasive Weed Optimization.\n" + Color.END)
    if not algorithm_type.strip():
        algorithm_type = 1
        print(f'\t Option 1 (DE) by default.')
    else:
        try:
            algorithm_type = int(algorithm_type)
            
            if algorithm_type > 3 or algorithm_type < 1:
                print(f'Error. Selected Algorithm ({algorithm_type}) is invalid.') 
                exit(1)

        except ValueError as e:
            print(f'Error: Invalid Input. {e}')
            exit(1)

    algorithm_type = int(algorithm_type)

    # Fitness Function Options:
    version_fitness = 1 # Sum of the squared errors (Default)

    ## ALGORITHM PARAMETERS SECTION ##
    print(Color.BOLD + f'\nAlgorithm parameters:\n' + Color.END)

    # Population size
    user_NPini = input(Color.BOLD + "Population size: " + Color.END)
    if not user_NPini.strip():
        user_NPini = 100
        print(f'Default population is {user_NPini}')
    else:
        try:
            user_NPini = int(user_NPini)
            
            if user_NPini <= 0:
                print(f'Error. Selected error ({user_NPini}) is invalid.') 
                exit(1)

        except ValueError as e:
            print(f'Error: Invalid Input. {e}')
            exit(1)

    # Max Iterations
    user_iter_max = input(Color.BOLD + "\nMax. iterations: " + Color.END)
    if not user_iter_max.strip():
        user_iter_max = 500
        print(f'Default iteration max is {user_iter_max}')
    else:
        try:
            user_iter_max = int(user_iter_max)
            
            if user_iter_max <= 0:
                print(f'Error. Selected error ({user_iter_max}) is invalid.') 
                exit(1)

        except ValueError as e:
            print(f'Error: Invalid Input. {e}')
            exit(1)

    return (id_cloud, err_dis, unif_noise, algorithm_type, version_fitness, user_NPini, user_iter_max, num_clouds)