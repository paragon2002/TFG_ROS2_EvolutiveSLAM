import pandas as pd
import os
import matplotlib.pyplot as plt
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
median_color = '#de1414'  # Rojo oscuro para la línea de la mediana

current_directory = os.path.dirname(__file__)
parent_directory = os.path.join(current_directory, '..')
filepath = os.path.join(parent_directory, 'errordata_sim.csv')

MAX_POS_ERROR = 0.25 # In meters
MAX_ORI_ERROR_1 = 8
MAX_ORI_ERROR_2 = 8
MAX_ORI_ERROR_3 = 8
MIN_CONVERGENCE_PERCENTAGE = 80
########################################################################

def getConvPerc(data_frame):
    pos_errors = data_frame['poserror_dist']
    ori_errors = data_frame[['orierror_1', 'orierror_2', 'orierror_3']]

    # Calcula la condición de convergencia
    converged = (pos_errors <= MAX_POS_ERROR) & (ori_errors <= [MAX_ORI_ERROR_1, MAX_ORI_ERROR_2, MAX_ORI_ERROR_3]).all(axis=1)

    total_samples = data_frame['id_cloud'].value_counts()

    convergence_counts = converged.groupby(data_frame['id_cloud']).sum()

    convergence_percentage = (convergence_counts / total_samples) * 100

    return convergence_percentage

def getDataStats(data_frame, col_id):

    ids = data_frame['id_cloud'].tolist()
    column_data = data_frame[col_id].tolist()

    pares_ordenados = sorted(zip(ids, column_data))

    df_pares_ordenados = pd.DataFrame(pares_ordenados, columns=['id_cloud', col_id])

    return df_pares_ordenados

def showBarCombined(dataframe, title, title_x, title_y, max_y_value=None,
                    custom_y_limit=None):

    # Función interna para graficar una parte del dataframe
    def plot_part(dataframe_part, title_suffix):
        plt.figure(figsize=(14, 6))
        
        # Obtener los datos para graficar
        ids = dataframe_part['id_cloud'].unique()
        bar_width = 0.2
        index = np.arange(len(ids))

        # Convertir los datos a arrays de numpy
        de_values = dataframe_part['DE'].values
        pso_values = dataframe_part['PSO'].values
        iwo_values = dataframe_part['IWO'].values

        # Crear las barras agrupadas
        plt.bar(index, de_values, bar_width, label='DE', color=all_colors[0])
        plt.bar(index + bar_width, pso_values, bar_width, label='PSO', color=all_colors[1])
        plt.bar(index + 2 * bar_width, iwo_values, bar_width, label='IWO', color=all_colors[2])

        if custom_y_limit:
            plt.axhline(y=custom_y_limit, color='gray', linestyle='--')

        if max_y_value:
            plt.ylim(0, max_y_value)

        # Configurar etiquetas y título
        plt.xlabel(title_x)
        plt.ylabel(title_y)
        plt.title(f"{title} - {title_suffix}")
        plt.xticks(index + bar_width, ids)
        plt.legend()

        # Mostrar la gráfica
        plt.tight_layout()
        plt.show()

    if len(dataframe['id_cloud'].unique()) > 22:

        dataframe_part1 = dataframe[dataframe['id_cloud'] <= 22]
        dataframe_part2 = dataframe[dataframe['id_cloud'] > 22]

        # Graficar la primera parte
        plot_part(dataframe_part1, "Primeras 22 Medidas")

        # Graficar la segunda parte
        plot_part(dataframe_part2, "Últimas 22 Medidas")

    else:
        plot_part(dataframe, "")


def showBoxPlotCombined(DE_df, PSO_df, IWO_df, title, title_x, title_y, max_y_value=None, custom_y_limit=None):

    # Función interna para graficar una parte del dataframe
    def plot_part(DE_part, PSO_part, IWO_part, title_suffix):
        plt.figure(figsize=(14, 6))
        
        # Obtener los datos para graficar
        ids = DE_part['id_cloud'].unique() # Aqui sirve cualquiera de los tres, todos tiene id_clouds del 1 al 44.

        # Crear las listas de datos para cada algoritmo
        data_de = [DE_part[DE_part['id_cloud'] == id]['Val'] for id in ids]
        data_pso = [PSO_part[PSO_part['id_cloud'] == id]['Val'] for id in ids]
        data_iwo = [IWO_part[IWO_part['id_cloud'] == id]['Val'] for id in ids]

        # Crear los boxplots agrupados
        positions_de = np.arange(len(ids)) * 3.0
        positions_pso = positions_de + 0.8
        positions_iwo = positions_de + 1.6

        plt.boxplot(data_de, positions=positions_de, widths=0.6, patch_artist=True, boxprops=dict(facecolor=all_colors[0]), medianprops=dict(color=median_color))
        plt.boxplot(data_pso, positions=positions_pso, widths=0.6, patch_artist=True, boxprops=dict(facecolor=all_colors[1]), medianprops=dict(color=median_color))
        plt.boxplot(data_iwo, positions=positions_iwo, widths=0.6, patch_artist=True, boxprops=dict(facecolor=all_colors[2]), medianprops=dict(color=median_color))

        if custom_y_limit:
            plt.axhline(y=custom_y_limit, color='gray', linestyle='--')

        # Añadir líneas de división
        for i in range(len(positions_de) - 1):
            plt.axvline(x=positions_de[i] + 2.2, color='black', linestyle='--', linewidth=0.5)

        if max_y_value:
            plt.ylim(0, max_y_value)

        # Configurar etiquetas y título
        plt.xlabel(title_x)
        plt.ylabel(title_y)
        plt.title(f"{title} - {title_suffix}")
        plt.xticks(positions_de + 0.8, ids)

        # Crear elementos de la leyenda
        legend_elements = [
            plt.Line2D([0], [0], color=all_colors[0], lw=4, label='DE'),
            plt.Line2D([0], [0], color=all_colors[1], lw=4, label='PSO'),
            plt.Line2D([0], [0], color=all_colors[2], lw=4, label='IWO')
        ]
        plt.legend(handles=legend_elements)


        # Mostrar la gráfica
        plt.tight_layout()
        plt.show()

    if len(DE_df['id_cloud'].unique()) > 22:
        DE_part_1 = DE_df[DE_df['id_cloud'] <= 22]
        DE_part_2 = DE_df[DE_df['id_cloud'] > 22]

        PSO_part_1 = PSO_df[PSO_df['id_cloud'] <= 22]
        PSO_part_2 = PSO_df[PSO_df['id_cloud'] > 22]

        IWO_part_1 = IWO_df[IWO_df['id_cloud'] <= 22]
        IWO_part_2 = IWO_df[IWO_df['id_cloud'] > 22]

        # Graficar la primera parte
        plot_part(DE_part_1, PSO_part_1, IWO_part_1, "Primeras 22 Medidas")

        # Graficar la segunda parte
        plot_part(DE_part_2, PSO_part_2, IWO_part_2, "Últimas 22 Medidas")

    else:
        plot_part(DE_df, PSO_df, IWO_df, "")

def main():
    data = pd.read_csv(filepath)

    de_data = data[data['algorithm'] == 1]
    pso_data = data[data['algorithm'] == 2]
    iwo_data = data[data['algorithm'] == 3]

    conv_perc_de = getConvPerc(de_data)
    conv_perc_pso = getConvPerc(pso_data)
    conv_perc_iwo = getConvPerc(iwo_data)

    time_avg_de = getDataStats(de_data, 'time')
    time_avg_pso = getDataStats(pso_data, 'time')
    time_avg_iwo = getDataStats(iwo_data, 'time')

    print(f"mean time DE: {np.mean(time_avg_de['time'])}")
    print(f"mean time PSO: {np.mean(time_avg_pso['time'])}")
    print(f"mean time IWO: {np.mean(time_avg_iwo['time'])}")

    it_avg_de = getDataStats(de_data, 'it')
    it_avg_pso = getDataStats(pso_data, 'it')
    it_avg_iwo = getDataStats(iwo_data, 'it')

    print("---------------------------------")
    print(f"mean its DE: {np.mean(it_avg_de['it'])}")
    print(f"mean its PSO: {np.mean(it_avg_pso['it'])}")
    print(f"mean its IWO: {np.mean(it_avg_iwo['it'])}")

    poserror_dist_de = getDataStats(de_data, 'poserror_dist')
    poserror_dist_pso = getDataStats(pso_data, 'poserror_dist')
    poserror_dist_iwo = getDataStats(iwo_data, 'poserror_dist')

    orierror1_stats_de = getDataStats(de_data, 'orierror_1')
    orierror1_stats_pso = getDataStats(pso_data, 'orierror_1')
    orierror1_stats_iwo = getDataStats(iwo_data, 'orierror_1')

    orierror2_stats_de = getDataStats(de_data, 'orierror_2')
    orierror2_stats_pso = getDataStats(pso_data, 'orierror_2')
    orierror2_stats_iwo = getDataStats(iwo_data, 'orierror_2')

    orierror3_stats_de = getDataStats(de_data, 'orierror_3')
    orierror3_stats_pso = getDataStats(pso_data, 'orierror_3')
    orierror3_stats_iwo = getDataStats(iwo_data, 'orierror_3')


    num_convergences_de = (conv_perc_de >= MIN_CONVERGENCE_PERCENTAGE).sum()
    num_convergences_pso = (conv_perc_pso >= MIN_CONVERGENCE_PERCENTAGE).sum()
    num_convergences_iwo = (conv_perc_iwo >= MIN_CONVERGENCE_PERCENTAGE).sum()
    print(Color.CYAN + f"DE Convergences: {num_convergences_de}/{conv_perc_de.count()}" + Color.END)
    print(Color.CYAN + f"PSO Convergences: {num_convergences_pso}/{conv_perc_pso.count()}" + Color.END)
    print(Color.CYAN + f"IWO Convergences: {num_convergences_iwo}/{conv_perc_iwo.count()}" + Color.END)

    title_x = 'Nube de puntos'

    title = f'Porcentaje de convergencia por nube de puntos por algoritmo. ({MAX_POS_ERROR}m, {MAX_ORI_ERROR_1}º, {MAX_ORI_ERROR_2}º, {MAX_ORI_ERROR_3}º)'
    title_y = 'Porcentaje de convergencia'
    combined_conv_perc = pd.DataFrame({'id_cloud': conv_perc_de.index, 'DE': conv_perc_de.values, 'PSO': conv_perc_pso.values, 'IWO': conv_perc_iwo.values})
    showBarCombined(combined_conv_perc, title, title_x, title_y, custom_y_limit=MIN_CONVERGENCE_PERCENTAGE)

    title = 'Tiempo de ejecución por nube de puntos por algoritmo'
    title_y = 'Tiempo de ejecución (segundos)'
    DE_time_avg = pd.DataFrame({'id_cloud': time_avg_de['id_cloud'], 'Val': time_avg_de['time']})
    PSO_time_avg = pd.DataFrame({'id_cloud': time_avg_pso['id_cloud'], 'Val': time_avg_pso['time']})
    IWO_time_avg = pd.DataFrame({'id_cloud': time_avg_iwo['id_cloud'], 'Val': time_avg_iwo['time']})
    showBoxPlotCombined(DE_time_avg, PSO_time_avg, IWO_time_avg, title, title_x, title_y, max_y_value=3500)

    title = 'Iteraciones por nube de puntos por algoritmo'
    title_y = 'Iteraciones'
    DE_it_avg = pd.DataFrame({'id_cloud': it_avg_de['id_cloud'], 'Val': it_avg_de['it']})
    PSO_it_avg = pd.DataFrame({'id_cloud': it_avg_pso['id_cloud'], 'Val': it_avg_pso['it']})
    IWO_it_avg = pd.DataFrame({'id_cloud': it_avg_iwo['id_cloud'], 'Val': it_avg_iwo['it']})
    showBoxPlotCombined(DE_it_avg, PSO_it_avg, IWO_it_avg, title, title_x, title_y, max_y_value=500)

    title = 'Error de posición por nube de puntos por algoritmo'
    title_y = 'Error de posición en metros'
    DE_pso_avg = pd.DataFrame({'id_cloud': poserror_dist_de['id_cloud'], 'Val': poserror_dist_de['poserror_dist']})
    PSO_pos_avg = pd.DataFrame({'id_cloud': poserror_dist_pso['id_cloud'], 'Val': poserror_dist_pso['poserror_dist']})
    IWO_pos_avg = pd.DataFrame({'id_cloud': poserror_dist_iwo['id_cloud'], 'Val': poserror_dist_iwo['poserror_dist']})
    showBoxPlotCombined(DE_pso_avg, PSO_pos_avg, IWO_pos_avg, title, title_x, title_y, custom_y_limit=MAX_POS_ERROR, max_y_value=12)

    title = 'Error de orientación (Cabeceo) por nube de puntos por algoritmo'
    title_y = 'Error de orientación en grados'
    DE_ori1_avg = pd.DataFrame({'id_cloud': orierror1_stats_de['id_cloud'], 'Val': orierror1_stats_de['orierror_1']})
    PSO_ori1_avg = pd.DataFrame({'id_cloud': orierror1_stats_pso['id_cloud'], 'Val': orierror1_stats_pso['orierror_1']})
    IWO_ori1_avg = pd.DataFrame({'id_cloud': orierror1_stats_iwo['id_cloud'], 'Val': orierror1_stats_iwo['orierror_1']})
    showBoxPlotCombined(DE_ori1_avg, PSO_ori1_avg, IWO_ori1_avg, title, title_x, title_y, custom_y_limit=MAX_ORI_ERROR_1, max_y_value=10)

    title = 'Error de orientación (Alabeo) por nube de puntos por algoritmo'
    title_y = 'Error de orientación en grados'
    DE_ori2_avg = pd.DataFrame({'id_cloud': orierror2_stats_de['id_cloud'], 'Val': orierror2_stats_de['orierror_2']})
    PSO_ori2_avg = pd.DataFrame({'id_cloud': orierror2_stats_pso['id_cloud'], 'Val': orierror2_stats_pso['orierror_2']})
    IWO_ori2_avg = pd.DataFrame({'id_cloud': orierror2_stats_iwo['id_cloud'], 'Val': orierror2_stats_iwo['orierror_2']})
    showBoxPlotCombined(DE_ori2_avg, PSO_ori2_avg, IWO_ori2_avg, title, title_x, title_y, custom_y_limit=MAX_ORI_ERROR_2, max_y_value=10)

    title = 'Error de orientación (Guiñada) por nube de puntos por algoritmo'
    title_y = 'Error de orientación en grados'
    DE_ori3_avg = pd.DataFrame({'id_cloud': orierror3_stats_de['id_cloud'], 'Val': orierror3_stats_de['orierror_3']})
    PSO_ori3_avg = pd.DataFrame({'id_cloud': orierror3_stats_pso['id_cloud'], 'Val': orierror3_stats_pso['orierror_3']})
    IWO_ori3_avg = pd.DataFrame({'id_cloud': orierror3_stats_iwo['id_cloud'], 'Val': orierror3_stats_iwo['orierror_3']})
    showBoxPlotCombined(DE_ori3_avg, PSO_ori3_avg, IWO_ori3_avg, title, title_x, title_y, custom_y_limit=MAX_ORI_ERROR_3, max_y_value=200)

if __name__ == '__main__':
    main()
