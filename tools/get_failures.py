import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ANSI color escape codes
class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# Naranja, azul, verde
all_colors = ['#E69F00', '#0072B2', '#009E73']

#home_route = os.path.expanduser("~")

current_directory = os.path.dirname(__file__)
parent_directory = os.path.join(current_directory, '..')
filepath = os.path.join(parent_directory, 'errordata.csv')

MAX_POS_ERROR = 0.25 # In meters
MAX_ORI_ERROR_1 = 8
MAX_ORI_ERROR_2 = 8
MAX_ORI_ERROR_3 = 8
MIN_CONVERGENCE_PERCENTAGE = 80

def main():
    algs = ["DE", "PSO", "IWO"]
    alg_id = 0

    data = pd.read_csv(filepath)

    data_de = data[data['algorithm'] == 1]

    data_pso = data[data['algorithm'] == 2]

    data_iwo = data[data['algorithm'] == 3]

    print('--------------------------')
    error_threshold = MAX_POS_ERROR
    for category in ['poserror_dist', 'orierror_1', 'orierror_2', 'orierror_3']:
        for dataset in [data_de, data_pso, data_iwo]:
            errors = dataset[dataset[category] > error_threshold].shape[0]
            total = dataset.shape[0]
            print(f"Fallos {category}, {algs[alg_id]}: {errors}/{total} ({round((errors/total)*100, 2)}%)")
            alg_id += 1
        alg_id = 0
        error_threshold = MAX_ORI_ERROR_1
        print('-----------------------------------------')

if __name__ == '__main__':
    main()
