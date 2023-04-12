from ngsolve import *
from random import random
import math


def give_random_magnetisation(mag_grid_func):
    assert len(mag_grid_func.vec) % 3 == 0, "The magnetisation vector data is not a multiple of three."
    num_points = len(mag_grid_func.vec) // 3
    for i in range(num_points):
        a,b,c = 2*random()-1, 2*random()-1, 2*random()-1
        size = math.sqrt(a*a + b*b + c*c)
        try:
            a,b,c = a/size, b/size, c/size
        except ZeroDivisionError:  # it is extremely unlikely, but possible, to have a=b=c=0. If this happens, use (1,0,0)
            a,b,c = 1, 0, 0
        mag_grid_func.vec[3*i] = a
        mag_grid_func.vec[3*i + 1] = b
        mag_grid_func.vec[3*i + 2] = c
    
    return mag_grid_func