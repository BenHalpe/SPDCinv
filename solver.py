from abc import ABC
from jax import jit
from spdc_inv.utils.utils import h_bar, eps0, c

import jax.numpy as np


import jax.random as random
import os

from jax.ops import index_update
from spdc_inv.data.utils import n_KTP_Kato, nz_MgCLN_Gayer
from spdc_inv.utils.utils import SFG_idler_wavelength
from spdc_inv.utils.defaults import REAL, IMAG
from typing import Dict

def Hermite_gauss(lam, refractive_index, W0, nx, ny, z, X, Y, coef=None):
    """
    Hermite Gauss in 2D

    Parameters
    ----------
    lam: wavelength
    refractive_index: refractive index
    W0: beam waists
    n, m: order of the HG beam
    z: the place in z to calculate for
    x,y: matrices of x and y
    coef

    Returns
    -------
    Hermite-Gaussian beam of order n,m in 2D
    """
    k = 2 * np.pi * refractive_index / lam
    z0 = np.pi * W0 ** 2 * refractive_index / lam  # Rayleigh range
    Wz = W0 * np.sqrt(1 + (z / z0) ** 2)  # w(z), the variation of the spot size

    invR = z / ((z ** 2) + (z0 ** 2))  # radius of curvature
    gouy = (nx + ny + 1)*np.arctan(z/z0)
    if coef is None:
        coefx = np.sqrt(np.sqrt(2/pi) / (2**nx * math.factorial(nx)))
        coefy = np.sqrt(np.sqrt(2/pi) / (2**ny * math.factorial(ny)))
        coef = coefx * coefy
    U = coef * \
        (W0/Wz) * np.exp(-(X**2 + Y**2) / Wz**2) * \
        HermiteP(nx, np.sqrt(2) * X / Wz) * \
        HermiteP(ny, np.sqrt(2) * Y / Wz) * \
        np.exp(-1j * (k * (X**2 + Y**2) / 2) * invR) * \
        np.exp(1j * gouy)

    return U


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


def HermiteP(n, x):
    """
    Hermite polynomial of rank n Hn(x)

    Parameters
    ----------
    n: order of the LG beam
    x: matrix of x

    Returns
    -------
    Hermite polynomial
    """
    if n == 0:
        return 1
    elif n == 1:
        return 2 * x
    else:
        return 2 * x * HermiteP(n - 1, x) - 2 * (n - 1) * HermiteP(n - 2, x)


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


class Beam(ABC):
    """
    A class that holds everything to do with a beam
    """
    def __init__(self,
                 lam: float,
                 ctype,
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

        self.lam          = lam
        self.n            = ctype(lam * 1e6, T, polarization)  # refractive index
        self.w            = 2 * np.pi * c / lam  # frequency
        self.k            = 2 * np.pi * ctype(lam * 1e6, T, polarization) / lam  # wave vector
        self.power        = power  # beam power


class Beam_profile(ABC):
    def __init__(
            self,
            pump_coeffs_real,
            pump_coeffs_imag,
            waist_pump,
            power_pump,
            x,
            y,
            dx,
            dy,
            max_mode1,
            max_mode2,
            pump_basis: str,
            lam_pump,
            refractive_index,
            z: float = 0.,
    ):


        self.x = x
        self.y = y
        self.z = z
        self.lam_pump          = lam_pump
        self.pump_basis        = pump_basis
        self.max_mode1         = max_mode1
        self.max_mode2         = max_mode2
        self.power             = power_pump
        self.crystal_dx        = dx
        self.crystal_dy        = dy
        self.refractive_index  = refractive_index

        self.pump_coeffs_real, \
        self.pump_coeffs_imag = pump_coeffs_real, pump_coeffs_imag
        self.waist_pump = waist_pump

        if self.pump_basis.lower() == 'lg':  # Laguerre-Gauss
            self.coef = np.zeros(len(waist_pump), dtype=np.float32)
            idx = 0
            for p in range(self.max_mode1):
                for l in range(-self.max_mode2, self.max_mode2 + 1):

                    self.coef = index_update(
                        self.coef, idx,
                        np.sqrt(2 * math.factorial(p) / (np.pi * math.factorial(p + np.abs(l))))
                    )

                    idx += 1

            self.E = self._profile_laguerre_gauss(pump_coeffs_real, pump_coeffs_imag, waist_pump)

        elif self.pump_basis.lower() == "hg":  # Hermite-Gauss
            self.coef = np.zeros(len(waist_pump), dtype=np.float32)
            idx = 0
            for nx in range(self.max_mode1):
                for ny in range(self.max_mode2):
                    self.coef = index_update(
                        self.coef, idx,
                        np.sqrt(np.sqrt(2 / pi) / (2 ** nx * math.factorial(nx))) *
                        np.sqrt(np.sqrt(2 / pi) / (2 ** ny * math.factorial(ny))))

                    idx += 1

            if not self.learn_pump:
                self.E = self._profile_hermite_gauss(pump_coeffs_real, pump_coeffs_imag, waist_pump)


    def create_profile(self, pump_coeffs_real, pump_coeffs_imag, waist_pump):
        # if self.learn_pump:
        #     if self.pump_basis.lower() == 'lg':  # Laguerre-Gauss
        #         if self.learn_pump_coeffs and self.learn_pump_waists:
        #             self.E = self._profile_laguerre_gauss(
        #                 pump_coeffs_real, pump_coeffs_imag, waist_pump
        #             )
        #         elif self.learn_pump_coeffs:
        #             self.E = self._profile_laguerre_gauss(
        #                 pump_coeffs_real, pump_coeffs_imag, self.waist_pump
        #             )
        #         else:
        #             self.E = self._profile_laguerre_gauss(
        #                 self.pump_coeffs_real, self.pump_coeffs_imag, waist_pump
        #             )

        #     elif self.pump_basis.lower() == 'hg':  # Hermite-Gauss
        #         if self.learn_pump_coeffs and self.learn_pump_waists:
        #             self.E = self._profile_hermite_gauss(
        #                 pump_coeffs_real, pump_coeffs_imag, waist_pump
        #             )
        #         elif self.learn_pump_coeffs:
        #             self.E = self._profile_hermite_gauss(
        #                 pump_coeffs_real, pump_coeffs_imag, self.waist_pump
        #             )
        #         else:
        #             self.E = self._profile_hermite_gauss(
        #                 self.pump_coeffs_real, self.pump_coeffs_imag, waist_pump
        #             )
        return

    def _profile_laguerre_gauss(
            self,
            pump_coeffs_real,
            pump_coeffs_imag,
            waist_pump
    ):
        coeffs = pump_coeffs_real + 1j * pump_coeffs_imag
        [X, Y] = np.meshgrid(self.x, self.y)
        pump_profile = 0.
        idx = 0
        for p in range(self.max_mode1):
            for l in range(-self.max_mode2, self.max_mode2 + 1):
                pump_profile += coeffs[idx] * \
                                Laguerre_gauss(self.lam_pump, self.refractive_index,
                                               waist_pump[idx] * 1e-5, l, p, self.z, X, Y, self.coef[idx])
                idx += 1

        pump_profile = fix_power(pump_profile, self.power, self.refractive_index,
                                 self.crystal_dx, self.crystal_dy)[np.newaxis, :, :]
        return pump_profile

    def _profile_hermite_gauss(
            self,
            pump_coeffs_real,
            pump_coeffs_imag,
            waist_pump
    ):

        coeffs = pump_coeffs_real + 1j * pump_coeffs_imag
        [X, Y] = np.meshgrid(self.x, self.y)
        pump_profile = 0.
        idx = 0
        for nx in range(self.max_mode1):
            for ny in range(self.max_mode2):
                pump_profile += coeffs[idx] * \
                                Hermite_gauss(self.lam_pump, self.refractive_index,
                                              waist_pump[idx] * 1e-5, nx, ny, self.z, X, Y, self.coef[idx])
                idx += 1

        pump_profile = fix_power(pump_profile, self.power, self.refractive_index,
                                 self.crystal_dx, self.crystal_dy)[np.newaxis, :, :]
        return pump_profile


class Crystal_hologram(ABC):
    def __init__(
            self,
            crystal_coeffs_real,
            crystal_coeffs_imag,
            r_scale,
            x,
            y,
            max_mode1,
            max_mode2,
            crystal_basis,
            lam_signal,
            refractive_index,
            z: float = 0.,
    ):

        self.x = x
        self.y = y
        self.z = z
        self.refractive_index     = refractive_index
        self.lam_signal           = lam_signal
        self.crystal_basis        = crystal_basis
        self.max_mode1 = max_mode1
        self.max_mode2 = max_mode2

        self.crystal_coeffs_real, \
        self.crystal_coeffs_imag = crystal_coeffs_real, crystal_coeffs_imag
        self.r_scale = r_scale



        if crystal_basis.lower() == 'ft':  # Fourier-Taylor
            if not self.learn_crystal:
                self.crystal_profile = self._profile_fourier_taylor(crystal_coeffs_real, crystal_coeffs_imag, r_scale)

        elif crystal_basis.lower() == 'fb':  # Fourier-Bessel

            [X, Y] = np.meshgrid(self.x, self.y)
            self.coef = np.zeros(len(r_scale), dtype=np.float32)
            idx = 0
            for p in range(self.max_mode1):
                for l in range(-self.max_mode2, self.max_mode2 + 1):
                    rad = np.sqrt(X ** 2 + Y ** 2) / (r_scale[idx] * 1e-5)
                    self.coef = index_update(
                        self.coef, idx,
                        sp.jv(0, sp.jn_zeros(0, p + 1)[-1] * rad)
                    )
                    idx += 1

            self.crystal_profile = self._profile_fourier_bessel(crystal_coeffs_real, crystal_coeffs_imag)

        elif crystal_basis.lower() == 'lg':  # Laguerre-Gauss

            self.coef = np.zeros(len(r_scale), dtype=np.float32)
            idx = 0
            for p in range(self.max_mode1):
                for l in range(-self.max_mode2, self.max_mode2 + 1):
                    self.coef = index_update(
                        self.coef, idx,
                        np.sqrt(2 * math.factorial(p) / (np.pi * math.factorial(p + np.abs(l))))
                    )
                    idx += 1

            self.crystal_profile = self._profile_laguerre_gauss(crystal_coeffs_real, crystal_coeffs_imag, r_scale)

        elif crystal_basis.lower() == 'hg':  # Hermite-Gauss

            self.coef = np.zeros(len(r_scale), dtype=np.float32)
            idx = 0
            for m in range(self.max_mode1):
                for n in range(self.max_mode2):
                    self.coef = index_update(
                        self.coef, idx,
                        np.sqrt(np.sqrt(2 / pi) / (2 ** m * math.factorial(m))) *
                        np.sqrt(np.sqrt(2 / pi) / (2 ** n * math.factorial(n)))
                    )

                    idx += 1

            if not self.learn_crystal:
                self.crystal_profile = self._profile_hermite_gauss(crystal_coeffs_real, crystal_coeffs_imag, r_scale)

    def create_profile(
            self,
            crystal_coeffs_real,
            crystal_coeffs_imag,
            r_scale,
    ):
        # if self.learn_crystal:
        #     if self.crystal_basis.lower() == 'ft':  # Fourier-Taylor
        #         if self.learn_crystal_coeffs and self.learn_crystal_waists:
        #             self.crystal_profile = self._profile_fourier_taylor(
        #                 crystal_coeffs_real, crystal_coeffs_imag, r_scale
        #             )
        #         elif self.learn_crystal_coeffs:
        #             self.crystal_profile = self._profile_fourier_taylor(
        #                 crystal_coeffs_real, crystal_coeffs_imag, self.r_scale
        #             )
        #         else:
        #             self.crystal_profile = self._profile_fourier_taylor(
        #                 self.crystal_coeffs_real, self.crystal_coeffs_imag, r_scale
        #             )

        #     elif self.crystal_basis.lower() == 'fb':  # Fourier-Bessel
        #         if self.learn_crystal_coeffs:
        #             self.crystal_profile = self._profile_fourier_bessel(
        #                 crystal_coeffs_real, crystal_coeffs_imag
        #             )
        #         else:
        #             self.crystal_profile = self._profile_fourier_bessel(
        #                 self.crystal_coeffs_real, self.crystal_coeffs_imag
        #             )

        #     elif self.crystal_basis.lower() == 'lg':  # Laguerre-Gauss
        #         if self.learn_crystal_coeffs and self.learn_crystal_waists:
        #             self.crystal_profile = self._profile_laguerre_gauss(
        #                 crystal_coeffs_real, crystal_coeffs_imag, r_scale
        #             )
        #         elif self.learn_crystal_coeffs:
        #             self.crystal_profile = self._profile_laguerre_gauss(
        #                 crystal_coeffs_real, crystal_coeffs_imag, self.r_scale
        #             )
        #         else:
        #             self.crystal_profile = self._profile_laguerre_gauss(
        #                 self.crystal_coeffs_real, self.crystal_coeffs_imag, r_scale
        #             )

        #     elif self.crystal_basis.lower() == 'hg':  # Hermite-Gauss
        #         if self.learn_crystal_coeffs and self.learn_crystal_waists:
        #             self.crystal_profile = self._profile_hermite_gauss(
        #                 crystal_coeffs_real, crystal_coeffs_imag, r_scale
        #             )
        #         elif self.learn_crystal_coeffs:
        #             self.crystal_profile = self._profile_hermite_gauss(
        #                 crystal_coeffs_real, crystal_coeffs_imag, self.r_scale
        #             )
        #         else:
        #             self.crystal_profile = self._profile_hermite_gauss(
        #                 self.crystal_coeffs_real, self.crystal_coeffs_imag, r_scale
        #             )
        return

    def _profile_fourier_taylor(
            self,
            crystal_coeffs_real,
            crystal_coeffs_imag,
            r_scale,
    ):
        coeffs = crystal_coeffs_real + 1j * crystal_coeffs_imag
        [X, Y] = np.meshgrid(self.x, self.y)
        phi_angle = np.arctan2(Y, X)
        crystal_profile = 0.
        idx = 0
        for p in range(self.max_mode1):
            for l in range(-self.max_mode2, self.max_mode2 + 1):
                rad = np.sqrt(X**2 + Y**2) / (r_scale[idx] * 1e-5)
                crystal_profile += coeffs[idx] * rad**p * np.exp(-rad**2) * np.exp(-1j * l * phi_angle)
                idx += 1

        return crystal_profile

    def _profile_fourier_bessel(
            self,
            crystal_coeffs_real,
            crystal_coeffs_imag,
    ):
        coeffs = crystal_coeffs_real + 1j * crystal_coeffs_imag
        [X, Y] = np.meshgrid(self.x, self.y)
        phi_angle = np.arctan2(Y, X)
        crystal_profile = 0.
        idx = 0
        for p in range(self.max_mode1):
            for l in range(-self.max_mode2, self.max_mode2 + 1):
                crystal_profile += coeffs[idx] * self.coef[idx] * np.exp(-1j * l * phi_angle)
                idx += 1

        return crystal_profile

    def _profile_laguerre_gauss(
            self,
            crystal_coeffs_real,
            crystal_coeffs_imag,
            r_scale,
    ):
        coeffs = crystal_coeffs_real + 1j * crystal_coeffs_imag
        [X, Y] = np.meshgrid(self.x, self.y)
        idx = 0
        crystal_profile = 0.
        for p in range(self.max_mode1):
            for l in range(-self.max_mode2, self.max_mode2 + 1):
                crystal_profile += coeffs[idx] * \
                                   Laguerre_gauss(self.lam_signal, self.refractive_index,
                                                  r_scale[idx] * 1e-5, l, p, self.z, X, Y, self.coef[idx])
                idx += 1

        return crystal_profile

    def _profile_hermite_gauss(
            self,
            crystal_coeffs_real,
            crystal_coeffs_imag,
            r_scale,
    ):
        coeffs = crystal_coeffs_real + 1j * crystal_coeffs_imag
        [X, Y] = np.meshgrid(self.x, self.y)
        idx = 0
        crystal_profile = 0.
        for m in range(self.max_mode1):
            for n in range(self.max_mode2):
                crystal_profile += coeffs[idx] * \
                                   Hermite_gauss(self.lam_signal, self.refractive_index,
                                                 r_scale[idx] * 1e-5, m, n, self.z, X, Y, self.coef[idx])

                idx += 1

        return crystal_profile


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


def PP_crystal_slab(
        delta_k,
        z,
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
    if crystal_profile is None:
        return np.sign(np.cos(np.abs(delta_k) * z))
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
                             np.sin(m * pi * DutyCycle) * 2 * np.cos(m * phase + m * np.abs(delta_k) * z)
            return poling
        else:
            return (2 / np.pi) * np.exp(1j * (np.abs(delta_k) * z)) * magnitude * np.exp(1j * phase)



class Interaction(ABC):
    """
    A class that represents the SPDC interaction process,
    on all of its physical parameters.
    """

    def __init__(
            self,
            pump_basis: str = 'LG',
            pump_max_mode1: int = 5,
            pump_max_mode2: int = 2,
            initial_pump_coefficient: str = 'uniform',
            custom_pump_coefficient: Dict[str, Dict[int, int]] = None,
            pump_coefficient_path: str = None,
            initial_pump_waist: str = 'waist_pump0',
            pump_waists_path: str = None,
            crystal_basis: str = 'LG',
            crystal_max_mode1: int = 5,
            crystal_max_mode2: int = 2,
            initial_crystal_coefficient: str = 'uniform',
            custom_crystal_coefficient: Dict[str, Dict[int, int]] = None,
            crystal_coefficient_path: str = None,
            initial_crystal_waist: str = 'r_scale0',
            crystal_waists_path: str = None,
            lam_pump: float = 405e-9,
            crystal_str: str = 'ktp',
            power_pump: float = 1e-3,
            waist_pump0: float = None,
            r_scale0: float = 40e-6,
            dx: float = 4e-6,
            dy: float = 4e-6,
            dz: float = 10e-6,
            maxX: float = 120e-6,
            maxY: float = 120e-6,
            maxZ: float = 1e-4,
            R: float = 0.1,
            Temperature: float = 50,
            pump_polarization: str = 'y',
            signal_polarization: str = 'y',
            idler_polarization: str = 'z',
            dk_offset: float = 1,
            power_signal: float = 1,
            power_idler: float = 1,
            key: np.array = None,

    ):
        """

        Parameters
        ----------
        pump_basis: Pump's construction basis method
                    Can be: LG (Laguerre-Gauss) / HG (Hermite-Gauss)
        pump_max_mode1: Maximum value of first mode of the 2D pump basis
        pump_max_mode2: Maximum value of second mode of the 2D pump basis
        initial_pump_coefficient: defines the initial distribution of coefficient-amplitudes for pump basis function
                                  can be: uniform- uniform distribution
                                          random- uniform distribution
                                          custom- as defined at custom_pump_coefficient
                                          load- will be loaded from np.arrays defined under path: pump_coefficient_path
                                                with names: parameters_pump_real.npy, parameters_pump_imag.npy
        pump_coefficient_path: path for loading waists for pump basis function
        custom_pump_coefficient: (dictionary) used only if initial_pump_coefficient=='custom'
                                 {'real': {indexes:coeffs}, 'imag': {indexes:coeffs}}.
        initial_pump_waist: defines the initial values of waists for pump basis function
                            can be: waist_pump0- will be set according to waist_pump0
                                    load- will be loaded from np.arrays defined under path: pump_waists_path
                                    with name: parameters_pump_waists.npy
        pump_waists_path: path for loading coefficient-amplitudes for pump basis function
        crystal_basis: Crystal's construction basis method
                       Can be:
                       None / FT (Fourier-Taylor) / FB (Fourier-Bessel) / LG (Laguerre-Gauss) / HG (Hermite-Gauss)
                       - if None, the crystal will contain NO hologram
        crystal_max_mode1: Maximum value of first mode of the 2D crystal basis
        crystal_max_mode2: Maximum value of second mode of the 2D crystal basis
        initial_crystal_coefficient: defines the initial distribution of coefficient-amplitudes for crystal basis function
                                     can be: uniform- uniform distribution
                                      random- uniform distribution
                                      custom- as defined at custom_crystal_coefficient
                                      load- will be loaded from np.arrays defined under path: crystal_coefficient_path
                                            with names: parameters_crystal_real.npy, parameters_crystal_imag.npy
        crystal_coefficient_path: path for loading coefficient-amplitudes for crystal basis function
        custom_crystal_coefficient: (dictionary) used only if initial_crystal_coefficient=='custom'
                                 {'real': {indexes:coeffs}, 'imag': {indexes:coeffs}}.
        initial_crystal_waist: defines the initial values of waists for crystal basis function
                               can be: r_scale0- will be set according to r_scale0
                                       load- will be loaded from np.arrays defined under path: crystal_waists_path
                                             with name: parameters_crystal_effective_waists.npy
        crystal_waists_path: path for loading waists for crystal basis function
        lam_pump: Pump wavelength
        crystal_str: Crystal type. Can be: KTP or MgCLN
        power_pump: Pump power [watt]
        waist_pump0: waists of the pump basis functions.
                     -- If None, waist_pump0 = sqrt(maxZ / self.pump_k)
        r_scale0: effective waists of the crystal basis functions.
                  -- If None, r_scale0 = waist_pump0
        dx: transverse resolution in x [m]
        dy: transverse resolution in y [m]
        dz: longitudinal resolution in z [m]
        maxX: Transverse cross-sectional size from the center of the crystal in x [m]
        maxY: Transverse cross-sectional size from the center of the crystal in y [m]
        maxZ: Crystal's length in z [m]
        R: distance to far-field screen [m]
        Temperature: crystal's temperature [Celsius Degrees]
        pump_polarization: Polarization of the pump beam
        signal_polarization: Polarization of the signal beam
        idler_polarization: Polarization of the idler beam
        dk_offset: delta_k offset
        power_signal: Signal power [watt]
        power_idler: Idler power [watt]
        key: Random key
        """

        self.lam_pump = lam_pump
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.x = np.arange(-maxX, maxX, dx)  # x axis, length 2*MaxX (transverse)
        self.y = np.arange(-maxY, maxY, dy)  # y axis, length 2*MaxY  (transverse)
        self.Nx = len(self.x)
        self.Ny = len(self.y)
        self.z = np.arange(-maxZ / 2, maxZ / 2, dz)  # z axis, length MaxZ (propagation)
        self.maxX = maxX
        self.maxY = maxY
        self.maxZ = maxZ
        self.R = R
        self.Temperature = Temperature
        self.dk_offset = dk_offset
        self.power_pump = power_pump
        self.power_signal = power_signal
        self.power_idler = power_idler
        self.lam_signal = 2 * lam_pump
        self.lam_idler = SFG_idler_wavelength(self.lam_pump, self.lam_signal)
        self.pump_polarization = pump_polarization
        self.signal_polarization = signal_polarization
        self.idler_polarization = idler_polarization
        self.key = key

        assert crystal_str.lower() in ['ktp', 'mgcln'], 'crystal must be either KTP or MgCLN'
        if crystal_str.lower() == 'ktp':
            self.ctype = n_KTP_Kato  # refractive index function
            self.pump_k = 2 * np.pi * n_KTP_Kato(lam_pump * 1e6, Temperature, pump_polarization) / lam_pump
            self.d33 = 16.9e-12  # nonlinear coefficient [meter/Volt]
        else:
            self.ctype = nz_MgCLN_Gayer  # refractive index function
            self.pump_k = 2 * np.pi * nz_MgCLN_Gayer(lam_pump * 1e6, Temperature) / lam_pump
            self.d33 = 23.4e-12  # [meter/Volt]
        # self.slab = PP_crystal_slab

        if waist_pump0 is None:
            self.waist_pump0 = np.sqrt(maxZ / self.pump_k)
        else:
            self.waist_pump0 = waist_pump0

        if r_scale0 is None:
            self.r_scale0 = self.waist_pump0
        else:
            self.r_scale0 = r_scale0

        assert pump_basis.lower() in ['lg', 'hg'], 'The beam structure is constructed as a combination ' \
                                                   'of LG or HG basis functions only'
        self.pump_basis = pump_basis
        self.pump_max_mode1 = pump_max_mode1
        self.pump_max_mode2 = pump_max_mode2
        self.initial_pump_coefficient = initial_pump_coefficient
        self.custom_pump_coefficient = custom_pump_coefficient
        self.pump_coefficient_path = pump_coefficient_path

        # number of modes for pump basis
        if pump_basis.lower() == 'lg':
            self.pump_n_modes1 = pump_max_mode1
            self.pump_n_modes2 = 2 * pump_max_mode2 + 1
        else:
            self.pump_n_modes1 = pump_max_mode1
            self.pump_n_modes2 = pump_max_mode2

        # Total number of pump modes
        self.pump_n_modes = self.pump_n_modes1 * self.pump_n_modes2

        self.initial_pump_waist = initial_pump_waist
        self.pump_waists_path = pump_waists_path

        self.crystal_basis = crystal_basis
        if crystal_basis:
            assert crystal_basis.lower() in ['ft', 'fb', 'lg', 'hg'], 'The crystal structure was constructed ' \
                                                                      'as a combination of FT, FB, LG or HG ' \
                                                                      'basis functions only'

            self.crystal_max_mode1 = crystal_max_mode1
            self.crystal_max_mode2 = crystal_max_mode2
            self.initial_crystal_coefficient = initial_crystal_coefficient
            self.custom_crystal_coefficient = custom_crystal_coefficient
            self.crystal_coefficient_path = crystal_coefficient_path

            # number of modes for crystal basis
            if crystal_basis.lower() in ['ft', 'fb', 'lg']:
                self.crystal_n_modes1 = crystal_max_mode1
                self.crystal_n_modes2 = 2 * crystal_max_mode2 + 1
            else:
                self.crystal_n_modes1 = crystal_max_mode1
                self.crystal_n_modes2 = crystal_max_mode2

            # Total number of crystal modes
            self.crystal_n_modes = self.crystal_n_modes1 * self.crystal_n_modes2

            self.initial_crystal_waist = initial_crystal_waist
            self.crystal_waists_path = crystal_waists_path

    def initial_pump_coefficients(
            self,
    ):

        if self.initial_pump_coefficient == "uniform":
            coeffs_real = np.ones(self.pump_n_modes, dtype=np.float32)
            coeffs_imag = np.ones(self.pump_n_modes, dtype=np.float32)

        elif self.initial_pump_coefficient == "random":

            self.key, pump_coeff_key = random.split(self.key)
            rand_real, rand_imag = random.split(pump_coeff_key)
            coeffs_real = random.normal(rand_real, (self.pump_n_modes,))
            coeffs_imag = random.normal(rand_imag, (self.pump_n_modes,))

        elif self.initial_pump_coefficient == "custom":
            assert self.custom_pump_coefficient, 'for custom method, pump basis coefficients and ' \
                                                 'indexes must be selected'
            coeffs_real = np.zeros(self.pump_n_modes, dtype=np.float32)
            coeffs_imag = np.zeros(self.pump_n_modes, dtype=np.float32)
            for index, coeff in self.custom_pump_coefficient[REAL].items():
                assert type(index) is int, f'index {index} must be int type'
                assert type(coeff) is float, f'coeff {coeff} must be float type'
                assert index < self.pump_n_modes, 'index for custom pump (real) initialization must be smaller ' \
                                                  'than total number of modes.' \
                                                  f'Got index {index} for total number of modes {self.pump_n_modes}'
                coeffs_real = index_update(coeffs_real, index, coeff)

            for index, coeff in self.custom_pump_coefficient[IMAG].items():
                assert type(index) is int, f'index {index} must be int type'
                assert type(coeff) is float, f'coeff {coeff} must be float type'
                assert index < self.pump_n_modes, 'index for custom pump (imag) initialization must be smaller ' \
                                                  'than total number of modes.' \
                                                  f'Got index {index} for total number of modes {self.pump_n_modes}'
                coeffs_imag = index_update(coeffs_imag, index, coeff)


        elif self.initial_pump_coefficient == "load":
            assert self.pump_coefficient_path, 'Path to pump coefficients must be defined'

            coeffs_real = np.load(os.path.join(self.pump_coefficient_path, 'parameters_pump_real.npy'))
            coeffs_imag = np.load(os.path.join(self.pump_coefficient_path, 'parameters_pump_imag.npy'))

        else:
            coeffs_real, coeffs_imag = None, None
            assert "ERROR: incompatible pump basis coefficients"

        normalization = np.sqrt(np.sum(np.abs(coeffs_real) ** 2 + np.abs(coeffs_imag) ** 2))
        coeffs_real = coeffs_real / normalization
        coeffs_imag = coeffs_imag / normalization

        return coeffs_real, coeffs_imag


    def initial_pump_waists(
            self,
    ):
        if self.initial_pump_waist == "waist_pump0":
            waist_pump = np.ones(self.pump_n_modes, dtype=np.float32) * self.waist_pump0 * 1e5

        elif self.initial_pump_waist == "load":
            assert self.pump_waists_path, 'Path to pump waists must be defined'

            waist_pump = np.load(os.path.join(self.pump_coefficient_path, "parameters_pump_waists.npy")) * 1e-1

        else:
            waist_pump = None
            assert "ERROR: incompatible pump basis waists"

        return waist_pump


    def initial_crystal_coefficients(
            self,
    ):
        if not self.crystal_basis:
            return None, None

        elif self.initial_crystal_coefficient == "uniform":
            coeffs_real = np.ones(self.crystal_n_modes, dtype=np.float32)
            coeffs_imag = np.ones(self.crystal_n_modes, dtype=np.float32)

        elif self.initial_crystal_coefficient == "random":

            self.key, crystal_coeff_key = random.split(self.key)
            rand_real, rand_imag = random.split(crystal_coeff_key)
            coeffs_real = random.normal(rand_real, (self.crystal_n_modes,))
            coeffs_imag = random.normal(rand_imag, (self.crystal_n_modes,))

        elif self.initial_crystal_coefficient == "custom":
            assert self.custom_crystal_coefficient, 'for custom method, crystal basis coefficients and ' \
                                                    'indexes must be selected'
            coeffs_real = np.zeros(self.crystal_n_modes, dtype=np.float32)
            coeffs_imag = np.zeros(self.crystal_n_modes, dtype=np.float32)
            for index, coeff in self.custom_crystal_coefficient[REAL].items():
                assert type(index) is int, f'index {index} must be int type'
                assert type(coeff) is float, f'coeff {coeff} must be float type'
                assert index < self.crystal_n_modes, 'index for custom crystal (real) initialization must be smaller ' \
                                                  'than total number of modes.' \
                                                  f'Got index {index} for total number of modes {self.crystal_n_modes}'
                coeffs_real = index_update(coeffs_real, index, coeff)

            for index, coeff in self.custom_crystal_coefficient[IMAG].items():
                assert type(index) is int, f'index {index} must be int type'
                assert type(coeff) is float, f'coeff {coeff} must be float type'
                assert index < self.crystal_n_modes, 'index for custom crystal (imag) initialization must be smaller ' \
                                                     'than total number of modes.' \
                                                     f'Got index {index} for total number of modes {self.crystal_n_modes}'
                coeffs_imag = index_update(coeffs_imag, index, coeff)

        elif self.initial_crystal_coefficient == "load":
            assert self.crystal_coefficient_path, 'Path to crystal coefficients must be defined'

            coeffs_real = np.load(os.path.join(self.crystal_coefficient_path, 'parameters_crystal_real.npy'))
            coeffs_imag = np.load(os.path.join(self.crystal_coefficient_path, 'parameters_crystal_imag.npy'))

        else:
            coeffs_real, coeffs_imag = None, None
            assert "ERROR: incompatible crystal basis coefficients"

        normalization = np.sqrt(np.sum(np.abs(coeffs_real) ** 2 + np.abs(coeffs_imag) ** 2))
        coeffs_real = coeffs_real / normalization
        coeffs_imag = coeffs_imag / normalization

        return coeffs_real, coeffs_imag


    def initial_crystal_waists(
            self,
    ):

        if not self.crystal_basis:
            return None

        if self.initial_crystal_waist == "r_scale0":
            r_scale = np.ones(self.crystal_n_modes, dtype=np.float32) * self.r_scale0 * 1e5

        elif self.initial_crystal_waist == "load":
            assert self.crystal_waists_path, 'Path to crystal waists must be defined'

            r_scale = np.load(os.path.join(self.crystal_waists_path, "parameters_crystal_effective_waists.npy")) * 1e-1

        else:
            r_scale = None
            assert "ERROR: incompatible crystal basis waists"

        return r_scale




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

########################## This is the solver ###########################################
def crystal_prop(
        pump_profile: Beam_profile,
        pump: Beam,
        signal_field: Field,
        idler_field: Field,
        vacuum_states,
        interaction: Interaction,
        poling_period, 
        N,
        crystal_hologram: Crystal_hologram,
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

    x  = interaction.x
    y  = interaction.y
    Nx = interaction.Nx
    Ny = interaction.Ny
    dz = interaction.dz


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

    for z in interaction.z:
        signal_out, signal_vac, idler_out, idler_vac = propagate_dz(
            pump_profile,
            x,
            y,
            z,
            dz,
            pump.k,
            signal_field.k,
            idler_field.k,
            signal_field.kappa,
            idler_field.kappa,
            poling_period,
            crystal_hologram,
            interaction.d33,
            signal_out,
            signal_vac,
            idler_out,
            idler_vac,
            infer
        )
    
    return signal_out, idler_out, idler_vac


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
        poling_period,
        crystal_hologram,
        interaction_d33,
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

    # crystal slab:
    PP = PP_crystal_slab(poling_period, z, crystal_hologram, inference=None)

    # coupled wave equations - split step
    # signal:
    dEs_out_dz = signal_field_kappa * interaction_d33 * PP * E_pump * np.conj(idler_vac)
    dEs_vac_dz = signal_field_kappa * interaction_d33 * PP * E_pump * np.conj(idler_out)
    signal_out = signal_out + dEs_out_dz * dz
    signal_vac = signal_vac + dEs_vac_dz * dz

    # idler:
    dEi_out_dz = idler_field_kappa * interaction_d33 * PP * E_pump * np.conj(signal_vac)
    dEi_vac_dz = idler_field_kappa * interaction_d33 * PP * E_pump * np.conj(signal_out)
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
