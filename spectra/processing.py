import logging
import numpy as np
from scipy import signal
from scipy import fftpack


def fourier_transform(f, dt, gamma):
    M = len(f)//2
    N = 2*M  # ensure even number of points (faster)
    c = dt/(2*np.pi)
    damped = damping(f[:N], dt, gamma)
    omega = fftpack.fftfreq(N, c)[:M]
    F = N*c*fftpack.ifft(damped, overwrite_x=True)[:M]
    return omega, F


def damping(signal, dt, gamma):
    '''damping on signal with decay rate gamma.'''
    t = np.arange(len(signal))*dt
    return signal*np.exp(-gamma*t)


def reduce_signal(signal, dt, reduction_factor=1):
    '''Decrease points of a discrete signal.
    '''
    if isinstance(reduction_factor, float):
        reduction_factor = int(reduction_factor)
    elif not isinstance(reduction_factor, int):
        raise TypeError("reduction_factor in reduce_signal must be integer.")

    if not isinstance(dt, (int, float)):
        raise TypeError("dt in reduce_signal must be float.")
    elif dt <= 0:
        raise ValueError("dt in reduce_signal must be positive.")

    if not isinstance(signal, np.ndarray):
        try:
            signal = np.array(signal)
        except:
            raise TypeError("signal in reduce_signal must be array-like.")

    return signal[::reduction_factor], dt*reduction_factor


def low_pass(f, dt, omega_max, order=7):
    if order <= 0:
        raise ValueError("order in low pass must be higher than 0.")
    b, a = signal.butter(order, omega_max*dt/np.pi, btype='low', analog=False)
    f_low_pass = signal.filtfilt(b, a, f)
    if np.isnan(np.sum(f_low_pass)):
        logging.warning("filter failed, trying a lower order.")
        f_low_pass = low_pass(f_low_pass, dt, omega_max, order - 1)
    return f_low_pass