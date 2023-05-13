from abc import ABC
from jax import jit
import math
import jax.numpy as np
import jax.random as random
import os
from jax.ops import index_update
from typing import Dict
from utils import *
from solver import *



print("Hello World!")

shape = Shape()
pump_lam = 405e-9
pump_waist = 40e-6

pump = Beam(lam=pump_lam, polarization="y", T=50, power=1e-3)
signal = Beam(lam=2*pump_lam, polarization="y", T=50, power=1)
idler = Beam(lam=SFG_idler_wavelength(pump.lam,signal.lam), polarization="z", T=50, power=1)
signal_field = Field(beam = signal,dx=shape.dx,dy=shape.dy,maxZ=shape.maxZ)
idler_field = Field(beam = idler,dx=shape.dx,dy=shape.dy,maxZ=shape.maxZ)


pump_profile = np.ones((shape.Nx,shape.Ny))
chi2 = np.ones((shape.Nz,shape.Nx,shape.Ny))


vacuum_states = np.ones((1,1,1))
# rand_key, subkey = random.split((1,2))
# initialize the vacuum and interaction fields
N=1
# vacuum_states = random.normal(subkey,(N, 2, 2, shape.Nx, shape.Ny))


A = crystal_prop(
        pump_profile, 
        pump,
        signal_field,
        idler_field,
        vacuum_states,
        chi2, 
        N,
        shape,
        infer=None,
        signal_init=None,
        idler_init=None,
        check_sol = False
)
print(len(A))
print(A[0])