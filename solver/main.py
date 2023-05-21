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
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import pandas as pd




print("Hello World!")

shape = Shape()
pump_lam = 405e-9
pump_waist = 40e-6
r_scale0 = 40e-6
dk_offset = 1
d33 = 16.9e-12  # nonlinear coefficient [meter/Volt] for ktp
# d33 = 23.4e-12  # nonlinear coefficient [meter/Volt] for mcGln

pump = Beam(lam=pump_lam, polarization="y", T=50, power=1e-3) 
signal = Beam(lam=2*pump_lam, polarization="y", T=50, power=1)
idler = Beam(lam=SFG_idler_wavelength(pump.lam,signal.lam), polarization="z", T=50, power=1)
signal_field = Field(beam = signal,dx=shape.dx,dy=shape.dy,maxZ=shape.maxZ)
idler_field = Field(beam = idler,dx=shape.dx,dy=shape.dy,maxZ=shape.maxZ)

# change to gauusian, and 
X,Y = np.meshgrid(shape.x,shape.y)
# pump_profile = Laguerre_gauss(pump_lam,pump.n,pump_waist,0,0,shape.z[0],X,Y)
#pump
max_mode1 = 1
max_mode2 = 1
idx = 2*max_mode2 + max_mode1
real_coeff = np.array([0,1,0])
img_coeff = np.array([1,0,1])
pump_waist = np.array([pump_waist]*idx)
pump_profile = profile_laguerre_gauss(real_coeff,img_coeff,pump_waist,shape,max_mode1,max_mode2,pump,mode="pump")
# crystal
max_mode1 = 1
max_mode2 = 1
idx = 2*max_mode2 + max_mode1
real_coeff = np.array([0,1,0])
img_coeff = np.array([1,0,1])
r_scale0 = np.array([r_scale0]*idx)
crystal_profile = profile_laguerre_gauss(real_coeff,img_coeff,r_scale0,shape,max_mode1,max_mode2,signal,mode="crystal")
print(crystal_profile.shape)
delta_k = pump.k - signal.k - idler.k  
poling_period = dk_offset * delta_k
PP = PP_crystal_slab(delta_k=delta_k, shape=shape, crystal_profile=crystal_profile, inference=None)


chi2 = PP*d33 

N=100
seed = 1701
key = random.PRNGKey(seed)
rand_key, subkey = random.split(key)
vacuum_states = random.normal(subkey,shape=(N,2,2,shape.Nx,shape.Ny))
# vacuum_states = np.ones(shape=(N,2,2,shape.Nx,shape.Ny))
# initialize the vacuum and interaction fields
# vacuum_states = random.normal(subkey,(N, 2, 2, shape.Nx, shape.Ny))


fig, ax = plt.subplots(dpi=150,subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, np.abs(pump_profile)**2, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
# fig = plt.figure(dpi=150)
# ax = fig.add_subplot(111)
# df = pd.DataFrame(data = np.abs(pump_profile)**2)
# ax = sns.heatmap(data=df)
# plt.show()
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
        check_sol = True
)
# print(len(A))
# print(A[0])

dict = {0:"signal out", 1:"signal vac", 2:"idler out", 3:"idler vac"}
for i in range(4):
        if (i%2==0):
                fig, ax = plt.subplots(dpi=150,subplot_kw={"projection": "3d"})
                surf = ax.plot_surface(X, Y, np.mean(np.abs(A[i])**2,axis=0), cmap=cm.coolwarm,linewidth=0, antialiased=False)
                fig.colorbar(surf, shrink=0.5, aspect=5)
                plt.title(f"{dict[i]}")
plt.show()
# for i in range(4):
#         fig = plt.figure(dpi=150)
#         ax = fig.add_subplot(111)
#         df = pd.DataFrame(data = np.abs(A[i][0])**2)
#         ax = sns.heatmap(data=df)
#         plt.show()



