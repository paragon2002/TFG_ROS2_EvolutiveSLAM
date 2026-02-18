from evloc.common_classes import Color
import csv
import os
import numpy as np

def save_error_data(id_cloud, algorithm_type, user_NPini, user_iter_max, D, F, CR, time, it, all_poserrors, all_orierrors, w, wdamp, c1, c2,
                    Smin, Smax, exponent, sigma_initial, sigma_final, stop_condition):
    """
    Saves the solution in the $HOME directory as a .csv file
    """

    home_route = os.path.expanduser("~")
    
    filename = "errordata.csv"
    filepath = os.path.join(home_route, filename)

    # Initializing CSV file with header if it doesn't exist
    if not os.path.exists(filepath):
        with open(filepath, mode='w', newline='') as archivo_csv:
            escritor_csv = csv.writer(archivo_csv)
            escritor_csv.writerow(['id_cloud', 'algorithm', 'NPini', 'iter_max', 'D', 'F', 'CR', 'w', 'wdamp',
                                    'c1', 'c2', 'Smin', 'Smax', 'exponent', 'sigma_initial', 'sigma_final', 'time', 'it', 'stop_condition',
                                    'poserror_dist', 'poserror_1', 'poserror_2', 'poserror_3', 'orierror_1', 'orierror_2', 'orierror_3'])


    # Escribir los datos en el archivo CSV
    with open(filepath, mode='a', newline='') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)

        poserror_dist = np.sqrt(all_poserrors[0]**2 + all_poserrors[1]**2 + all_poserrors[2]**2)

        if algorithm_type == 1:
            escritor_csv.writerow([id_cloud] + [algorithm_type] + [user_NPini] + [user_iter_max] + [D] + [F] + [CR] + [None] + [None] + [None] + [None]
                                  + [None] + [None] + [None] + [None] + [None]  + [time] + [it] + [stop_condition] + [poserror_dist]
                                  + all_poserrors + all_orierrors)
        if algorithm_type == 2:
            escritor_csv.writerow([id_cloud] + [algorithm_type] + [user_NPini] + [user_iter_max] + [D] + [None] + [None] + [w] + [wdamp] + [c1] + [c2]
                                  + [None] + [None] + [None] + [None] + [None] + [time] + [it] + [stop_condition] + [poserror_dist]
                                  + all_poserrors + all_orierrors)
        if algorithm_type == 3:
            escritor_csv.writerow([id_cloud] + [algorithm_type] + [user_NPini] + [user_iter_max] + [D] + [None] + [None] + [None] + [None] + [None] + [None] 
                                  + [Smin] + [Smax] + [exponent] + [sigma_initial] + [sigma_final] + [time] + [it] + [stop_condition] + [poserror_dist]
                                  + all_poserrors + all_orierrors)


    print("\n"+Color.BOLD + "SUMMARY:" + Color.END)
    print(f"id_cloud: {id_cloud}")
    print(f"algorithm_type: {algorithm_type}")
    print(f"user_NPini: {user_NPini}")
    print(f"user_iter_max: {user_iter_max}")
    print(f"D: {D}")
    print(f"F: {F}")
    print(f"CR: {CR}")
    print(f"Smin: {Smin}")
    print(f"Smax: {Smax}")
    print(f"exponent: {exponent}")
    print(f"sigma_initial: {sigma_initial}")
    print(f"sigma_final: {sigma_final}")
    print(f"time: {time}s")
    print(f"it: {it}")
    print(f"stop_condition: {stop_condition}")
    print(f"poserror: {all_poserrors}")
    print(f"poserror_dist: {poserror_dist}")
    print(f"orierror: {all_orierrors}\n")

    print(Color.BOLD + f"Data Saved in {filepath}" + Color.END + "\n")