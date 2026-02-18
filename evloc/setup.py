from setuptools import find_packages, setup
import os
import glob

package_name = 'evloc'

# Función para copiar todos los archivos de una carpeta preservando la estructura
def package_files(directory):
    paths = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            paths.append(os.path.join(root, f))
    return paths

# Archivos sueltos en real y simulation
real_files = [
    'resources/real/groundtruth_data.csv',
    'resources/real/map_global_ori.ply'
]

sim_files = [
    'resources/simulation/groundtruth_data.csv',
    'resources/simulation/map_global.pcd'
]

# Todos los archivos dentro de local_clouds
real_clouds = package_files('resources/real/local_clouds')
sim_clouds = package_files('resources/simulation/local_clouds')

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # Archivos sueltos de real y simulation
        ('share/evloc/resources/real', real_files),
        ('share/evloc/resources/simulation', sim_files),

        # Carpeta local_clouds dentro de real y simulation
        ('share/evloc/resources/real/local_clouds', real_clouds),
        ('share/evloc/resources/simulation/local_clouds', sim_clouds),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='GonzaloVega',
    maintainer_email='g.vega.2020@alumnos.urjc.es',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "evloc_node = evloc.evloc_node:main"
        ],
    },
)
