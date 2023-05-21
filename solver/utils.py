from abc import ABC
from jax import jit
import math

import jax.numpy as np


import jax.random as random
import os

from jax.ops import index_update
from typing import Dict

# Constants:
pi      = np.pi
c       = 2.99792458e8  # speed of light [meter/sec]
eps0    = 8.854187817e-12  # vacuum permittivity [Farad/meter]
h_bar   = 1.054571800e-34  # [m^2 kg / s], taken from http://physics.nist.gov/cgi-bin/cuu/Value?hbar|search_for=planck

def n_KTP_Kato(
        lam: float,
        T: float,
        ax: str,
):
    """
    Refractive index for KTP, based on K. Kato

    Parameters
    ----------
    lam: wavelength (lambda) [um]
    T: Temperature [Celsius Degrees]
    ax: polarization

    Returns
    -------
    n: Refractive index

    """
    assert ax in ['z', 'y'], 'polarization must be either z or y'
    dT = (T - 20)
    if ax == "z":
        n_no_T_dep = np.sqrt(4.59423 + 0.06206 / (lam ** 2 - 0.04763) + 110.80672 / (lam ** 2 - 86.12171))
        dn         = (0.9221 / lam ** 3 - 2.9220 / lam ** 2 + 3.6677 / lam - 0.1897) * 1e-5 * dT
    if ax == "y":
        n_no_T_dep = np.sqrt(3.45018 + 0.04341 / (lam ** 2 - 0.04597) + 16.98825 / (lam ** 2 - 39.43799))
        dn         = (0.1997 / lam ** 3 - 0.4063 / lam ** 2 + 0.5154 / lam + 0.5425) * 1e-5 * dT
    n           = n_no_T_dep + dn
    return n



class Beam(ABC):
    """
    A class that holds everything to do with a beam
    """
    def __init__(self,
                 lam: float,
                 polarization: str,
                 T: float,
                 power: float = 0):

        """

        Parameters
        ----------
        lam: beam's wavelength
        ctype: function that holds crystal type fo calculating refractive index
        polarization: Polarization of the beam
        T: crystal's temperature [Celsius Degrees]
        power: beam power [watt]
        """
        ctype = self.ctype = n_KTP_Kato
        self.lam          = lam
        self.n            = ctype(lam * 1e6, T, polarization)  # refractive index
        self.w            = 2 * np.pi * c / lam  # frequency
        self.k            = 2 * np.pi * ctype(lam * 1e6, T, polarization) / lam  # wave vector
        self.power        = power  # beam power


class Field(ABC):
    """
    A class that holds everything to do with the interaction values of a given beam
    vac   - corresponding vacuum state coefficient
    kappa - coupling constant
    k     - wave vector
    """
    def __init__(
            self,
            beam,
            dx,
            dy,
            maxZ
    ):
        """

        Parameters
        ----------
        beam: A class that holds everything to do with a beam
        dx: transverse resolution in x [m]
        dy: transverse resolution in y [m]
        maxZ: Crystal's length in z [m]
        """

        self.vac   = np.sqrt(h_bar * beam.w / (2 * eps0 * beam.n ** 2 * dx * dy * maxZ))
        self.kappa = 2 * 1j * beam.w ** 2 / (beam.k * c ** 2)
        self.k     = beam.k

class Shape():
    """
    A class that holds everything to do with the dimensions
    """
    def __init__(
            self,
            dx: float = 4e-6,
            dy: float = 4e-6,
            dz: float = 10e-7,
            maxX: float = 120e-6,
            maxY: float = 120e-6,
            maxZ: float = 1e-4,
    ):


        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.x = np.arange(-maxX, maxX, dx)  # x axis, length 2*MaxX (transverse)
        self.y = np.arange(-maxY, maxY, dy)  # y axis, length 2*MaxY  (transverse)
        self.Nx = len(self.x)
        self.Ny = len(self.y)
        self.z = np.arange(-maxZ / 2, maxZ / 2, dz)  # z axis, length MaxZ (propagation)
        self.Nz = len(self.z)
        self.maxX = maxX
        self.maxY = maxY
        self.maxZ = maxZ






SFG_idler_wavelength    = lambda lambda_p, lambda_s: lambda_p * lambda_s / (lambda_s - lambda_p)
I                       = lambda A, n: 2 * n * eps0 * c * np.abs(A) ** 2  # Intensity
Power2D                 = lambda A, n, dx, dy: np.sum(I(A, n)) * dx * dy
def fix_power(
        A,
        power,
        n,
        dx,
        dy
):
    """
    The function takes a field A and normalizes in to have the power indicated

    Parameters
    ----------
    A
    power
    n
    dx
    dy

    Returns
    -------

    """
    output = A * np.sqrt(power) / np.sqrt(Power2D(A, n, dx, dy))
    return output

def check_equations(
        pump_profile,
        z,
        dx,
        dy,
        dz,
        pump_k,
        signal_field_k,
        idler_field_k,
        signal_field_kappa,
        idler_field_kappa,
        chi2,
        signal_out,
        signal_vac,
        idler_out,
        idler_vac,
):
    """
    Check if the second harmonic coupled wave equations holds and return the MSE of each 
    equation writen as a price function

    Parameters
    ----------
    pump_profile
    z: the position in the z direction
    dx: transverse resolution in x [m]
    dy: transverse resolution in y [m]
    dz: longitudinal resolution in z [m]
    pump_k: pump k vector
    signal_field_k: signal k vector
    idler_field_k: field k vector
    signal_field_kappa: signal kappa
    idler_field_kappa: idler kappa
    chi2: scaler array of chi2 in the crystal
    signal_out: tuple ofcurrent signal profile and previous profile
    signal_vac: tuple ofcurrent signal vacuum state profile and previous profile
    idler_out: tuple ofcurrent idler profile and previous profile
    idler_vac: tuple ofcurrent idler vacuum state profile and previous profile

    Returns
    -------
    MSE = (m1,m2,m3,m4) where mi is the MSE of the i'th equation
    """
    deltaK = pump_k - signal_field_k - idler_field_k
    d_dz = lambda E: (E[1] - E[0])/dz
    dd_dxx = lambda E: (E[1][:,2:,1:-1]+E[1][:,:-2,1:-1]-2*E[1][:,1:-1,1:-1])/dx**2
    dd_dyy = lambda E: (E[1][:,1:-1,2:]+E[1][:,1:-1,:-2]-2*E[1][:,1:-1,1:-1])/dy**2
    trans_laplasian=  lambda E: dd_dxx(E)+dd_dyy(E)
    f = lambda E1,k1,kapa1,E2: (1j*d_dz(E1)[:,1:-1,1:-1] + trans_laplasian(E1)/(2*k1) 
         - kapa1*chi2[1:-1,1:-1]*pump_profile[1:-1,1:-1]*np.exp(-1j*deltaK*z)*np.conj(E2[1][:,1:-1,1:-1]))
    
    m1 = np.mean(np.abs(f(idler_out,idler_field_k,idler_field_kappa,signal_vac))**2)
    m2 = np.mean(np.abs(f(idler_vac,idler_field_k,idler_field_kappa,signal_out))**2)
    m3 = np.mean(np.abs(f(signal_out,signal_field_k,signal_field_kappa,idler_vac))**2)
    m4 = np.mean(np.abs(f(signal_vac,signal_field_k,signal_field_kappa,idler_out))**2)

    return (m1,m2,m3,m4)


def LaguerreP(p, l, x):
    """
    Generalized Laguerre polynomial of rank p,l L_p^|l|(x)

    Parameters
    ----------
    l, p: order of the LG beam
    x: matrix of x

    Returns
    -------
    Generalized Laguerre polynomial
    """
    if p == 0:
        return 1
    elif p == 1:
        return 1 + np.abs(l)-x
    else:
        return ((2*p-1+np.abs(l)-x)*LaguerreP(p-1, l, x) - (p-1+np.abs(l))*LaguerreP(p-2, l, x))/p




def Laguerre_gauss(lam, refractive_index, W0, l, p, z, x, y, coef=None):
    """
    Laguerre Gauss in 2D

    Parameters
    ----------
    lam: wavelength
    refractive_index: refractive index
    W0: beam waists
    l, p: order of the LG beam
    z: the place in z to calculate for
    x,y: matrices of x and y
    coef

    Returns
    -------
    Laguerre-Gaussian beam of order l,p in 2D
    """
    k = 2 * np.pi * refractive_index / lam
    z0 = np.pi * W0 ** 2 * refractive_index / lam  # Rayleigh range
    Wz = W0 * np.sqrt(1 + (z / z0) ** 2)  # w(z), the variation of the spot size
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    invR = z / ((z ** 2) + (z0 ** 2))  # radius of curvature
    gouy = (np.abs(l)+2*p+1)*np.arctan(z/z0)
    if coef is None:
        coef = np.sqrt(2*math.factorial(p)/(np.pi * math.factorial(p + np.abs(l))))

    U = coef * \
        (W0/Wz)*(r*np.sqrt(2)/Wz)**(np.abs(l)) * \
        np.exp(-r**2 / Wz**2) * \
        LaguerreP(p, l, 2 * r**2 / Wz**2) * \
        np.exp(-1j * (k * r**2 / 2) * invR) * \
        np.exp(-1j * l * phi) * \
        np.exp(1j * gouy)
    return U

def profile_laguerre_gauss(
            pump_coeffs_real,
            pump_coeffs_imag,
            waist_pump,
            shape,
            max_mode1,
            max_mode2,
            beam,
            mode
    ):
        coeffs = pump_coeffs_real + 1j * pump_coeffs_imag
        if mode == "pump":
            [X, Y] = np.meshgrid(shape.x, shape.y)
            Z = 0
        elif mode == "crystal":
            [Z, X, Y] = np.meshgrid(shape.z,shape.x, shape.y, indexing='ij')
        
        pump_profile = np.zeros((shape.Nx, shape.Ny))
        idx = 0
        for p in range(max_mode1):
            for l in range(-max_mode2, max_mode2 + 1):
                pump_profile += coeffs[idx] * \
                                Laguerre_gauss(beam.lam, beam.n,
                                               waist_pump[idx] , l, p, Z, X, Y)
                idx += 1

        pump_profile = fix_power(pump_profile, beam.power, beam.n,
                                 shape.dx, shape.dy)
        return pump_profile



def PP_crystal_slab(
        delta_k,
        shape,
        crystal_profile,
        inference=None
):
    """
    Periodically poled crystal slab.
    create the crystal slab at point z in the crystal, for poling period 2pi/delta_k

    Parameters
    ----------
    delta_k: k mismatch
    z: longitudinal point for generating poling pattern
    crystal_profile: Crystal 3D hologram (if None, ignore)
    inference: (True/False) if in inference mode, we include more coefficients in the poling
                description for better validation

    Returns Periodically poled crystal slab at point z
    -------

    """
    [Z, X, Y] = np.meshgrid(shape.z,shape.x, shape.y, indexing='ij')
    if crystal_profile is None:
        return np.sign(np.cos(np.abs(delta_k) * Z))
    else:
        magnitude = np.abs(crystal_profile)
        phase = np.angle(crystal_profile)
        if inference is not None:
            max_order_fourier = 20
            poling = 0
            magnitude = magnitude / magnitude.max()
            DutyCycle = np.arcsin(magnitude) / np.pi
            for m in range(max_order_fourier):
                if m == 0:
                    poling = poling + 2 * DutyCycle - 1
                else:
                    poling = poling + (2 / (m * np.pi)) * \
                             np.sin(m * pi * DutyCycle) * 2 * np.cos(m * phase + m * np.abs(delta_k) * Z)
            return poling
        else:
            return (2 / np.pi) * np.exp(1j * (np.abs(delta_k) * Z)) * magnitude * np.exp(1j * phase)



# interaction_params = {
#         'pump_basis': 'HG',
#         'pump_max_mode1': 1,
#         'pump_max_mode2': 1,
#         'initial_pump_coefficient': 'custom',
#         'custom_pump_coefficient': {REAL: {0: 0., 1: 0., 2: 0., 3: 0., 4: 1., 5: 0., 6: 0., 7: 0., 8: 0.},
#                                     IMAG: {0: 0., 1: 0., 2: 0.}},
#         'pump_coefficient_path': None,
#         'initial_pump_waist': 'waist_pump0',
#         'pump_waists_path': None,
#         'crystal_basis': 'LG',
#         'crystal_max_mode1': 10,
#         'crystal_max_mode2': 4,
#         'initial_crystal_coefficient': 'custom',
#         'custom_crystal_coefficient': {REAL: {4: 1.}, IMAG: {0: 0., 1: 0., 2: 0.}},
#         'crystal_coefficient_path': None,
#         'initial_crystal_waist': 'r_scale0',
#         'crystal_waists_path': None,
#         'lam_pump': 405e-9,
#         'crystal_str': 'ktp',
#         'power_pump': 1e-3,
#         'waist_pump0': 40e-6,
#         'r_scale0': 40e-6,
#         'dx': 4e-6,
#         'dy': 4e-6,
#         'dz': 10e-6,
#         'maxX': 180e-6,
#         'maxY': 180e-6,
#         'maxZ': 1e-3,
#         'R': 0.1,
#         'Temperature': 50,
#         'pump_polarization': 'y',
#         'signal_polarization': 'y',
#         'idler_polarization': 'z',
#         'dk_offset': 1.,
#         'power_signal': 1.,
#         'power_idler': 1.,
#     }
