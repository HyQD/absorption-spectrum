import numpy as np
from scipy import linalg

from .processing import reduce_signal


class FourierPade:
    def __init__(self, dipole_moment, dt, reduction_factor=1):
        mu, self._dt = reduce_signal(dipole_moment, dt, reduction_factor=reduction_factor)
        self._compute_coefficients(mu)

    def _compute_coefficients(self, mu):
        '''Following convention of Bruner et al.'''
        M = (len(mu) - 1)//2
        N = M*2 + 1
        mu_ = np.flip(mu[:M+1])

        # create polynomial for denominator
        b = np.ones(M+1)
        d = -mu[M+1:N]
        G = linalg.toeplitz(mu[M:2*M], mu_[:-1])
        b[1:] = linalg.solve(G, d, overwrite_a=True, overwrite_b=True)
        self._Q = np.polynomial.Polynomial(b)

        # create polynomial for numerator
        a = np.zeros(M+1)
        for k in range(M+1):
            a[k] = np.sum(b[:k+1]*mu_[M-k:])
        self._P = np.polynomial.Polynomial(a)

    def __call__(self, omega, gamma):
        z = self.z(omega, gamma)
        M_M = self.P(z)/self.Q(z)
        return self.dt/(2*np.pi)*M_M

    def z(self, omega, gamma=0):
        theta = (1j*omega - gamma)*self.dt
        return np.exp(theta)

    def absorption_spectrum(self, omega, gamma, F):
        return _spectrum(self.__call__(omega, gamma), omega, F, normalize=False)

    def real_poles(self):
        z = self.Q.roots()
        z = z[z.imag > 0] # remove duplicates on real axis
        return np.log(z).imag/self.dt

    @property
    def P(self):
        return self._P

    @property
    def Q(self):
        return self._Q

    @property
    def dt(self):
        return self._dt