from ngsolve import *
from random import random
import math
import numpy as np

mu = 1
lam = 1


def strain(u):
    '''
    Returns the total strain from (grad(u) + grad(u)^T) / 2.
    '''
    return Sym(grad(u))


def strain_el(m, u):
    return strain(u) - strain_m(m, u)


def strain_m(m, u):
    return None


def stress(strain):
    '''
    Returns the stress associated with (the isotropic) Hooke's law from a given strain.
    '''
    return 2*mu*strain + lam*Trace(strain)*Id(3)
