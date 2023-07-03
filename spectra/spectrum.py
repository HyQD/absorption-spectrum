import numpy as np
from scipy.constants import speed_of_light, physical_constants

from .processing import fourier_transform


# speed of light in atomic units
SPEED_OF_LIGHT = speed_of_light/physical_constants["atomic unit of velocity"][0]


def absorption_spectrum(mu_x, mu_y, mu_z, omega, F, normalize=True):
    S = omega*np.imag((mu_x + mu_y + mu_x)/F)
    if normalize:
        return S/S.max()
    else:
        return 4*np.pi/(3*SPEED_OF_LIGHT)*S


def discrete_spectrum(dipole_moment, dt, F, gamma, normalize=True):
    '''Absorption spectrum using a discrete Fourier transform
    '''
    omega, mu_x = fourier_transform(dipole_moment[0], dt, gamma)
    omega, mu_y = fourier_transform(dipole_moment[1], dt, gamma)
    omega, mu_z = fourier_transform(dipole_moment[2], dt, gamma)
    return omega, absorption_spectrum(mu_x, mu_y, mu_z, omega, F, normalize)


def pade_spectrum(MM_x, MM_y, MM_z, omega, F, gamma, normalize=True):
    mu_x = MM_x(omega, gamma)
    mu_y = MM_y(omega, gamma)
    mu_z = MM_z(omega, gamma)
    return absorption_spectrum(mu_x, mu_y, mu_z, omega, F, normalize)