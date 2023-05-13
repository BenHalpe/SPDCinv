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
            dz: float = 10e-6,
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
    dd_dxx = lambda E: (E[1][:,2:,1:-1]+E[1][:,:-2,1:-1]-2*E[1][:,1,:-1,1:-1])/dx**2
    dd_dyy = lambda E: (E[1][:,1:-1,2:]+E[1][:,1:-1,:-2]-2*E[1][:,1:-1,1,:-1])/dy**2
    trans_laplasian=  lambda E: dd_dxx(E)+dd_dyy(E)
    f = lambda E1,k1,kapa1,E2: (1j*d_dz(E1) + trans_laplasian(E1)/(2*k1) 
         - kapa1*chi2[1:-1,1:-1]*pump_profile[1:-1,1:-1]*np.exp(-1j*deltaK*z)*E2[1][:,1:-1,1:-1].conj())
    
    m1 = np.mean(np.abs(f(idler_out,idler_field_k,idler_field_kappa,signal_vac))**2)
    m2 = np.mean(np.abs(f(idler_vac,idler_field_k,idler_field_kappa,signal_out))**2)
    m3 = np.mean(np.abs(f(signal_out,signal_field_k,signal_field_kappa,idler_vac))**2)
    m4 = np.mean(np.abs(f(signal_vac,signal_field_k,signal_field_kappa,idler_out))**2)

    return (m1,m2,m3,m4)

    