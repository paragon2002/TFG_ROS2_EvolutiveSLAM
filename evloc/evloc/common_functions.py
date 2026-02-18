import numpy as np
import open3d as o3d


def add_noise_to_pc(cloud, err_dis, unif_noise):
    """
    Returns a point cloud with the noise added.
    """

    sensnoise_PCmat = cloud.points
    dist_3d = distance_pc_to_point(cloud.points, [0, 0, 0])
    err_m = np.random.randn(dist_3d.size)  # Generating random samples from a normal distribution
    err_level = dist_3d * err_dis
    new_dist_3d = dist_3d + (err_m * err_level)

    CONTAMINATION_LEVEL = unif_noise
    CORRECT_MEASUREMENTS = 1 - CONTAMINATION_LEVEL
    err_m2 = (np.random.rand(*new_dist_3d.shape) * 0.5) + 0.25
    err_cont = new_dist_3d * err_m2
    err_m3 = np.random.rand(*new_dist_3d.shape)
    err_m3 = (err_m3 <= CORRECT_MEASUREMENTS).astype(int)

    err_m4 = np.ones_like(err_m3) - err_m3
    new_dist_3d = err_m3 * new_dist_3d + err_m4 * err_cont  # The new scan is formed

    noise_mat = (new_dist_3d / dist_3d).T * sensnoise_PCmat

    noise_mat_no_nan = np.nan_to_num(noise_mat)
    # Create an Open3D PointCloud
    new_cloud = o3d.geometry.PointCloud()
    new_cloud.points = o3d.utility.Vector3dVector(noise_mat_no_nan)

    return new_cloud


def costfunction3d(dist_scan, dist_map, version_fitness, err_dis):
    """
    Descripción: función de ajuste que se optimiza mediante el filtro de localización global.
    Las mediciones láser de la exploración actual se comparan con las mediciones láser
    en el vecino más cercano sobre el mapa real para calcular un valor de costo
    para una estimación. Este valor de costo se minimizará para obtener la solución
    del problema de localización global.

    En el caso de considerar la norma L2, la contribución al
    error en cada medida es el cuadrado de la diferencia entre la medida estimada y la real,
    dividido por la varianza multiplicada por 2, para aplicar criterios de convergencia.
    """
    if version_fitness == 1:
        error = np.sum(np.sum(((dist_scan - dist_map)**2) / (1 + 2 * (err_dis * dist_scan)**2)))

    # En el caso de considerar la norma L1, la contribución al
    # error en cada medida es la diferencia entre la medida estimada y la real, en valor
    # absoluto, dividido por la desviación típica, para aplicar criterios de convergencia.
    if version_fitness == 2:
        error = np.sum(np.sum(np.abs(dist_scan - dist_map) / (1 + err_dis * dist_scan)))

    return error


def distance_pc_to_point(cloud_mat, point):
    """
    Distance from each point of a pointcloud to a point

    -cloud_mat: Open3D PointCloud object
    -point: (x y z)
    - dist=horizontal vector with each PCpoint distance to point
    """
    cloud_array = np.asarray(cloud_mat)

    dist_mat = np.zeros((1, cloud_array.shape[0]))

    for i in range(cloud_array.shape[0]):
        dist_mat[0, i] = np.sqrt((cloud_array[i, 0] - point[0]) ** 2 +
                                 (cloud_array[i, 1] - point[1]) ** 2 +
                                 (cloud_array[i, 2] - point[2]) ** 2)

    return dist_mat


def spatial_rotation(point, p):
    """
    spatialTransformation
    The point gets transform by multiplying by the coordinate frame of the
    first scan according to the parameters
    Inputs:
      point: Horizontal vector nx3
      p:  Vertical vector 6x1 (rad)
    Outputs:
      transformed: Horizontal vector nx3 with the point transformed in the
      space
    """

    cAlpha = np.cos(p[3])
    sAlpha = np.sin(p[3])
    cBeta = np.cos(p[4])
    sBeta = np.sin(p[4])
    cGamma = np.cos(p[5])
    sGamma = np.sin(p[5])

    rotation_matrix = np.array([
        [cBeta * cGamma, -cBeta * sGamma, sBeta],
        [cAlpha * sGamma + cGamma * sAlpha * sBeta, cAlpha * cGamma - sAlpha * sBeta * sGamma, -cBeta * sAlpha],
        [sAlpha * sGamma - cAlpha * cGamma * sBeta, cGamma * sAlpha + cAlpha * sBeta * sGamma, cAlpha * cBeta]
    ])

    transformed = np.dot(point, rotation_matrix.T) + np.array([p[0], p[1], p[2]])

    return transformed