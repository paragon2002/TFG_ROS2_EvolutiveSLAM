import open3d as o3d
from evloc.common_functions import spatial_rotation

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


class Solution:
    """
    Class where the solution of the algorithm is stored.
    it: Number of iterations to converge.
    timediff: Time elapsed during the execution.
    all_estimates: List containing the best estimated pose of every iteration
    pos_error: Position error in meters
    ori_error: Orientation error in degrees.
    map_global: Global map
    real_scan: Local map
    """
    def __init__(self, it, timediff, all_estimates, all_poserrors, all_orierrors, map_global, real_scan, stop_condition):
        self.it = it
        self.time = timediff
        self.all_pose_estimates = all_estimates
        self.all_poserrors = all_poserrors
        self.all_orierrors = all_orierrors
        self.map = map_global
        self.stop_condition = stop_condition
    
class Algorithm:
    """
    Class that stores the parameters of the algorithm that will be used
    """
    def __init__(self, type=None, NPini=None, iter_max=None, D=None, F=None, CR=None, w=None, wdamp=None, c1=None, c2=None,
                 Smin=None, Smax=None, exponent=None, sigma_initial=None, sigma_final=None):
        self.type = type
        self.NPini = NPini
        self.iter_max = iter_max
        self.D = D
        self.F = F
        self.CR = CR
        self.w = w
        self.wdamp = wdamp
        self.c1 = c1
        self.c2 = c2
        self.Smin = Smin 
        self.Smax = Smax
        self.exponent = exponent
        self.sigma_initial = sigma_initial 
        self.sigma_final = sigma_final