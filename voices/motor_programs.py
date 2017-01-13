import numpy as np


class LinearMotorProgram(object):

    def __init__(self, ndim=4):
        # generate random initial state
        self.x = np.random.randn(ndim)
        self.W = np.zeros([ndim, ndim])
        self.Wout = np.zeros([2, ndim])
        self.ndim = ndim

    def get_num_params(self):
        return self.ndim**2

    def set_params(self, p):
        num_expected = self.ndim**2 + self.ndim*2
        assert len(p) == num_expected, "Wrong number of parameters, expected %d, got %d" % (num_expected, len(p))
        i = self.ndim**2
        pW = p[:i]
        pWout = p[i:]

        self.W = pW.reshape([self.ndim, self.ndim])

        # sparsify matrix
        M = np.random.rand(self.ndim, self.ndim)
        self.W[M < 0.75] = 0.

        # set spectral radius of matrix
        rescale_matrix(self.W, spectral_radius=1.5)

        self.Wout = pWout.reshape([2, self.ndim])

    def next(self):

        self.x = np.dot(self.W, self.x)
        out = np.dot(self.Wout, self.x)
        # apply a sigmoid output nonlinearity multiplied by a gain
        u = 100. / (1 + np.exp(-out[0]))
        k = 1e2*np.tanh(out[1])

        return np.array([u, k])


def rescale_matrix(W, spectral_radius=1.0):
    """ Rescale the given matrix W (in place) until it's spectral radius is less than or equal to the given value. """

    max_decrease_factor = 0.90
    min_decrease_factor = 0.99
    evals,evecs = np.linalg.eig(W)
    initial_eigenvalue = np.max(np.abs(evals))
    max_eigenvalue = initial_eigenvalue
    while max_eigenvalue > spectral_radius:
        #inverse distance to 1.0, a number between 0.0 (far) and 1.0 (close)
        d = 1.0 - ((max_eigenvalue - spectral_radius) / abs(initial_eigenvalue - spectral_radius))

        decrease_factor = max_decrease_factor + d*(min_decrease_factor-max_decrease_factor)
        W *= decrease_factor
        evals,evecs = np.linalg.eig(W)
        max_eigenvalue = np.max(np.abs(evals))