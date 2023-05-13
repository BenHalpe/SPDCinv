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



def crystal_prop(
        pump_profile, 
        pump: Beam,
        signal_field: Field,
        idler_field: Field,
        vacuum_states,
        chi2, 
        N,
        shape,
        infer=None,
        signal_init=None,
        idler_init=None
):
    """
    Crystal propagation
    propagate through crystal using split step Fourier for 4 fields: signal, idler and two vacuum states

    Parameters
    ----------
    pump_profile: electromagnetic pump beam profile
    pump: A class that holds everything to do with the pump beam
    signal_field: A class that holds everything to do with the interaction values of the signal beam
    idler_field: A class that holds everything to do with the interaction values of the idler beam
    vacuum_states: The vacuum and interaction fields
    interaction: A class that represents the SPDC interaction process, on all of its physical parameters.
    poling_period: Poling period (dk_offset * delta_k) :=
      # = interaction.dk_offset * self.delta_k, 
      delta_k= pump.k - signal.k - idler.k  
      # phase  mismatch
    N: number of vacuum_state elements
    crystal_hologram: 3D crystal hologram
    infer: (True/False) if in inference mode, we include more coefficients in the poling
                description for better validation
    signal_init: initial signal profile. If None, initiate to zero
    idler_init: initial idler profile. If None, initiate to zero

    Returns: the interacting fields at the end of interaction medium
    -------

    """

    x  = shape.x
    y  = shape.y
    z  = shape.z
    Nx = shape.Nx
    Ny = shape.Ny
    Nz = shape.Nz
    dz = shape.dz


    if signal_init is None:
        signal_out = np.zeros([N, Nx, Ny])
    else:
        assert len(signal_init.shape) == 3
        assert signal_init.shape[0] == N
        assert signal_init.shape[1] == Nx
        assert signal_init.shape[2] == Ny
        signal_out = signal_init

    if idler_init is None:
        idler_out = np.zeros([N, Nx, Ny])
    else:
        assert len(idler_init.shape) == 3
        assert idler_init.shape[0] == N
        assert idler_init.shape[1] == Nx
        assert idler_init.shape[2] == Ny
        idler_out = idler_init


    signal_vac = signal_field.vac * (vacuum_states[:, 0, 0] + 1j * vacuum_states[:, 0, 1]) / np.sqrt(2)
    idler_vac  = idler_field.vac * (vacuum_states[:, 1, 0] + 1j * vacuum_states[:, 1, 1]) / np.sqrt(2)

    for i in range(shape.Nz):
        signal_out, signal_vac, idler_out, idler_vac = propagate_dz(
            pump_profile,
            x,
            y,
            z[i],
            dz,
            pump.k,
            signal_field.k,
            idler_field.k,
            signal_field.kappa,
            idler_field.kappa,
            chi2[:,:,i],
            signal_out,
            signal_vac,
            idler_out,
            idler_vac,
            infer
        )
    
    return signal_out, signal_vac, idler_out, idler_vac


@jit
def propagate_dz(
        pump_profile,
        x,
        y,
        z,
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
        infer=None,
):
    """
    Single step of crystal propagation
    single split step Fourier for 4 fields: signal, idler and two vacuum states

    Parameters
    ----------
    pump_profile
    x:  x axis, length 2*MaxX (transverse)
    y:  y axis, length 2*MaxY  (transverse)
    z:  z axis, length MaxZ (propagation)
    dz: longitudinal resolution in z [m]
    pump_k: pump k vector
    signal_field_k: signal k vector
    idler_field_k: field k vector
    signal_field_kappa: signal kappa
    idler_field_kappa: idler kappa
    poling_period: poling period
    crystal_hologram: Crystal 3D hologram (if None, ignore)
    interaction_d33: nonlinear coefficient [meter/Volt]
    signal_out: current signal profile
    signal_vac: current signal vacuum state profile
    idler_out: current idler profile
    idler_vac: current idler vacuum state profile
    infer: (True/False) if in inference mode, we include more coefficients in the poling
                description for better validation

    Returns
    -------

    """

    # pump beam:
    E_pump = propagate(pump_profile, x, y, pump_k, z) * np.exp(-1j * pump_k * z)


    # coupled wave equations - split step
    # signal:
    dEs_out_dz = signal_field_kappa * chi2 * E_pump * np.conj(idler_vac)
    dEs_vac_dz = signal_field_kappa * chi2 * E_pump * np.conj(idler_out)
    signal_out = signal_out + dEs_out_dz * dz
    signal_vac = signal_vac + dEs_vac_dz * dz

    # idler:
    dEi_out_dz = idler_field_kappa * chi2 * E_pump * np.conj(signal_vac)
    dEi_vac_dz = idler_field_kappa * chi2 * E_pump * np.conj(signal_out)
    idler_out = idler_out + dEi_out_dz * dz
    idler_vac = idler_vac + dEi_vac_dz * dz

    # propagate
    signal_out = propagate(signal_out, x, y, signal_field_k, dz) * np.exp(-1j * signal_field_k * dz)
    signal_vac = propagate(signal_vac, x, y, signal_field_k, dz) * np.exp(-1j * signal_field_k * dz)
    idler_out = propagate(idler_out, x, y, idler_field_k, dz) * np.exp(-1j * idler_field_k * dz)
    idler_vac = propagate(idler_vac, x, y, idler_field_k, dz) * np.exp(-1j * idler_field_k * dz)

    return signal_out, signal_vac, idler_out, idler_vac


@jit
def propagate(A, x, y, k, dz):
    """
    Free Space propagation using the free space transfer function,
    (two  dimensional), according to Saleh
    Using CGS, or MKS, Boyd 2nd eddition

    Parameters
    ----------
    A: electromagnetic beam profile
    x,y: spatial vectors
    k: wave vector
    dz: The distance to propagate

    Returns the propagated field
    -------

    """
    dx      = np.abs(x[1] - x[0])
    dy      = np.abs(y[1] - y[0])

    # define the fourier vectors
    X, Y    = np.meshgrid(x, y, indexing='ij')
    KX      = 2 * np.pi * (X / dx) / (np.size(X, 1) * dx)
    KY      = 2 * np.pi * (Y / dy) / (np.size(Y, 1) * dy)

    # The Free space transfer function of propagation, using the Fresnel approximation
    # (from "Engineering optics with matlab"/ing-ChungPoon&TaegeunKim):
    H_w = np.exp(-1j * dz * (np.square(KX) + np.square(KY)) / (2 * k))
    H_w = np.fft.ifftshift(H_w)

    # Fourier Transform: move to k-space
    G = np.fft.fft2(A)  # The two-dimensional discrete Fourier transform (DFT) of A.

    # propoagte in the fourier space
    F = np.multiply(G, H_w)

    # inverse Fourier Transform: go back to real space
    Eout = np.fft.ifft2(F)  # [in real space]. E1 is the two-dimensional INVERSE discrete Fourier transform (DFT) of F1

    return Eout


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

print("Hello World!")

shape = Shape()
pump_lam = 405e-9
pump_waist = 40e-6

pump = Beam(lam=pump_lam, polarization="y", T=50, power=1e-3)
signal = Beam(lam=2*pump_lam, polarization="y", T=50, power=1)
idler = Beam(lam=SFG_idler_wavelength(pump.lam,signal.lam), polarization="z", T=50, power=1)
signal_field = Field(beam = signal,dx=shape.dx,dy=shape.dy,maxZ=shape.maxZ)
idler_field = Field(beam = idler,dx=shape.dx,dy=shape.dy,maxZ=shape.maxZ)

X,Y = np.meshgrid(shape.x,shape.y)

pump_profile = np.exp((-X**2-Y**2)/pump_waist**2) 
pump_profile = fix_power(pump_profile,pump.power,pump.n,shape.dx,shape.dy)
chi2 = np.ones((shape.Nx,shape.Ny,shape.Nz))


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
        idler_init=None
)
print(len(A))
print(A[0])