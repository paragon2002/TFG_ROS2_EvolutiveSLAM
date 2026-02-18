from ament_index_python.packages import get_package_share_directory
import os

PACKAGE_PATH = os.path.join(get_package_share_directory('evloc'), 'resources')

GROUNDTRUTH_FILE_PATH = f"{PACKAGE_PATH}/groundtruth_data.csv"
LOCAL_CLOUDS_FOLDER = f"{PACKAGE_PATH}/local_clouds"

DOWN_SAMPLING_FACTOR_GLOBAL = 0.004     # factor de downsampling para mapa, hay que reducir puntos en ambas nubes
DOWN_SAMPLING_FACTOR = 0.01             # factor de downsampling para scan
POP_RATIO = 0.01

def test():
    print("TEST")