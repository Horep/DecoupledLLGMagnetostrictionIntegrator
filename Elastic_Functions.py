from ngsolve import *
from random import random
import math
import numpy as np

mu = 1
lam = 1


def Strain_el(u):
    return Sym(grad(u))


def Stress(strain):
    return 2*mu*strain + lam*Trace(strain)*Id(3)
