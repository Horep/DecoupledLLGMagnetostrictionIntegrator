from ngsolve import *
from random import random
import math
import numpy as np

mu = 1
lam = 1

def Stress(u):
    strain = Sym(grad(u))
    return 2*mu*strain + lam*Trace(strain)*Id(3)
