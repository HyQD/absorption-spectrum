import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

from .processing import low_pass
from .pade import FourierPade


class DipoleMoment:
    '''Approximation to the dipole moment.'''
    def __init__(self, cutoff_frequency=np.inf, error_tol=1e-3):
        self.cutoff_frequency = cutoff_frequency
        self.error_tol = error_tol # Error tolerance for the fit
        self._set_default_parameters()

    def _set_default_parameters(self):
        self.fit_verification_ratio = 0.75
        self.kmeans_tol = 1e-6 # K-means tolerance parameter
        self.cutoff_buffer = 2 # Frequency estimation buffer
        self.max_points_pade = 5e3 # Max number of time points for Pade
        self.unstable_cycles = 1

        # unknown parameters
        self._frequencies = np.empty(0)
        self._A = np.empty(0)
        self._B = np.empty(0)
        self._D = 0
        self._error = np.nan

    def __call__(self, t):
        mu = np.zeros_like(t, dtype=np.float64)
        for n in range(self.N):
            phi = self.frequencies[n]*t
            mu += self.A[n]*np.cos(phi) 
            mu += self.B[n]*np.sin(phi) 
        return mu + self.D

    def use_filter(self):
        return self.cutoff_frequency is not np.inf

    @property
    def frequencies(self):
        '''Bohr frequencies.'''
        return self._frequencies

    @property
    def A(self):
        '''Cosine coefficients.'''
        return self._A

    @property
    def B(self):
        '''Sine coefficients.'''
        return self._B

    @property
    def D(self):
        '''Constant.'''
        return self._D

    @property
    def N(self):
        '''Number of frequencies.'''
        return len(self.frequencies)

    @property
    def error(self):
        '''Validation error of the fit.'''
        return self._error

    @property
    def cutoff_frequency(self):
        return self._cutoff_frequency

    @cutoff_frequency.setter 
    def cutoff_frequency(self, cutoff_frequency):
        if not isinstance(cutoff_frequency, (int, float)):
            raise TypeError("cut-off frequency must be scalar.")
        elif not isinstance(cutoff_frequency, (int, float)):
            raise ValueError("cut-off frequency must be positive.")
        self._cutoff_frequency = cutoff_frequency

    @property
    def max_points_pade(self):
        return self._max_points_pade

    @max_points_pade.setter 
    def max_points_pade(self, max_points_pade):
        if not isinstance(max_points_pade, (int, float)):
            raise TypeError("Max number of points used in Padé must be int.")
        elif max_points_pade <= 0:
            raise ValueError("Max number of points used in Padé must be positive.")
        self._max_points_pade = int(max_points_pade)

    def fit(self, dipole_moment, time_points):
        self.estimate_frequencies(dipole_moment, time_points)
        self.optimize_coefficients(dipole_moment, time_points)
        return self.error < self.error_tol

    def estimate_frequencies(self, dipole_moment, time_points):
        pade = self._create_pade(dipole_moment, time_points)
        prospective_frequecies = pade.real_poles()
        features = self._cluster_features(pade, prospective_frequecies)
        kmeans = KMeans(2, tol=self.kmeans_tol).fit(features)

        # Determine centroid closest to (0, 0) & Remove larger frequencies
        centroids = np.sum(kmeans.cluster_centers_, axis=1)
        estimated = prospective_frequecies[kmeans.labels_ == np.argmin(centroids)]
        max_frequency = self.cutoff_frequency + self.cutoff_buffer
        self._frequencies = estimated[estimated < max_frequency]
        self._A = np.zeros(self.N)
        self._B = np.zeros(self.N)
    
    def _create_pade(self, dipole_moment, time_points):
        reduction = len(dipole_moment)//self.max_points_pade + 1
        dt = time_points[1] - time_points[0]
        if self.use_filter() and np.pi/dt < self.cutoff_frequency*reduction:
            logging.warning("Time step is too large to capture cutoff-frequency.")
        return FourierPade(dipole_moment, dt, reduction_factor=reduction)

    def _cluster_features(self, pade, prospective_frequecies):
        '''position vectors normalized to [0, 1]'''
        z = pade.z(prospective_frequecies)
        Q = np.abs(pade.Q(z))
        I = np.abs(pade.P(z))/Q
        Q, I = np.log10(Q), np.log10(I)
    
        X = np.zeros((len(prospective_frequecies), 2))
        X[:, 0] = 1 - (I - I.min())/(I.max() - I.min())
        X[:, 1] = (Q - Q.min())/(Q.max() - Q.min())
        return X

    def optimize_coefficients(self, dipole_moment, time_points):
        if self.use_filter():
            dipole_moment = self._filter(dipole_moment, time_points)
        Nt = len(dipole_moment)
        Mt = int(Nt*self.fit_verification_ratio)
        mu = dipole_moment[:Mt]
    
        # Left-hand side of the coefficient linear system:
        t = time_points[:Mt]
        LHS = np.ones((Mt, 2*self.N + 1))
        for i, omega in enumerate(self.frequencies):
            LHS[:, i] = np.cos(omega*t)
            LHS[:, self.N+i] = np.sin(omega*t)
        
        # Solve linear system:
        U, s, VT = np.linalg.svd(LHS, full_matrices=False)
        c = VT.T@np.diag(1/s)@U.T@mu
        self._A = c[:self.N]
        self._B = c[self.N:-1]
        self._D = c[-1]
        self._error = self._compute_error(dipole_moment[Mt:Nt], time_points[Mt:Nt])

    def _number_of_unstable_points(self, dt):
        cycle_length = 2*np.pi/self.cutoff_frequency
        return int(self.unstable_cycles*cycle_length/dt)

    def _cut_unstable_data(self, dipole_moment, dt):
        N = len(dipole_moment)
        unstable = self._number_of_unstable_points(dt)
        if unstable > N//4:
            problem = "End of dipole moment is discarded when filtering due to instabilities."
            logging.warning(f"{problem} Current dipole length is insufficent to account for this.")
            return dipole_moment[:N//2]
        return dipole_moment[:-unstable]

    def _filter(self, dipole_moment, time_points):
        # remove unstable last part if filtered
        dt = time_points[1] - time_points[0]
        filtered = low_pass(dipole_moment, dt, self.cutoff_frequency)
        return self._cut_unstable_data(filtered, dt)

    def _compute_error(self, dipole_moment, time_points):
        '''Self-evaluation of the error.'''
        computed = self.__call__(time_points)
        return 1 - r2_score(dipole_moment, computed)

    def save(self, filename):
        '''Save to npz-file.'''
        data = {}
        data["frequencies"] = self._frequencies
        data["A"] = self._A
        data["B"] = self._B
        data["D"] = self._D
        data["cutoff_frequency"] = self._cutoff_frequency
        data["error"] = self._error
        if not filename.endswith(".npz"):
            filename += ".npz"
        np.savez(filename, **data)

    @classmethod
    def create(cls, frequencies, A, B, D, cutoff_frequency=np.inf, error=np.nan):
        '''Create object from with known frequencies and coefficients.'''
        mu = DipoleMoment(cutoff_frequency=float(cutoff_frequency))
        mu._frequencies = np.array(frequencies)
        mu._A = np.array(A)
        mu._B = np.array(B)
        mu._D = float(D)
        mu._error = float(error)
        return mu

    @classmethod
    def load(cls, filename):
        '''Create object from numpy zip-file.'''
        if not filename.endswith(".npz"):
            filename += ".npz"
        data = np.load(filename, allow_pickle=True)
        return cls.create(**data)


class BroadbandDipole(DipoleMoment):
    def _set_default_parameters(self):
        super()._set_default_parameters()
        # LASSO parameters
        self.alpha = 1e-12        # Penalty parameter
        self.lasso_tol = 1e-5     # Absolute tolerance for the optimization
        self.lasso_iter = 5000    # Maximum number of iterations

    def __call__(self, t):
        mu = np.zeros_like(t, dtype=np.float64)
        for n in range(self.N):
            phi = self.frequencies[n]*t
            mu += self.B[n]*np.sin(phi) 
        return mu + self.D

    def transform(self, omega, gamma):
        '''Damped Fourier transform using t > 0.'''
        LR = np.zeros_like(omega, dtype=np.complex128)
        omega2 = (omega + 1j*gamma)**2
        phi = 1j*omega - gamma
        for n in range(self.N):
            rr = omega2 - self.frequencies[n]**2
            LR -= self.B[n]*self.frequencies[n]/rr
        return LR/(2*np.pi)

    def optimize_coefficients(self, dipole_moment, time_points):
        if self.use_filter():
            dipole_moment = self._filter(dipole_moment, time_points)
        Nt = len(dipole_moment)
        Mt = int(Nt*self.fit_verification_ratio)
        mu = dipole_moment[:Mt]
        
        # Left-hand side of the coefficient linear system:
        t = time_points[:Mt]
        LHS = np.zeros((Mt, self.N))
        for i in range(self.N):
            LHS[:, i] = np.sin(self.frequencies[i]*t)
        
        # Solve linear system:
        mean = mu.mean()
        scale = np.abs(mu - mean).max()
        lasso = Lasso(
                    alpha=self.alpha,
                    positive=True, 
                    tol=self.lasso_tol,
                    copy_X=False, 
                    selection="random", 
                    max_iter=self.lasso_iter,
        )
        lasso.fit(LHS/scale, (mu - mean)/scale)

        self._A = np.zeros(self.N)
        self._B = lasso.coef_[:]
        self._D = scale*lasso.intercept_ + mean
        self._error = self._compute_error(dipole_moment[Mt:Nt], time_points[Mt:Nt])