import scipy.spatial.distance as distance
import numpy as np


class DiffMap:
    def __init__(self, X, t, epsilon, sigma):
        self.X = X
        self.t = t
        self.epsilon = epsilon
        self.sigma = sigma

    def compute_eigen(self):
        K = distance.pdist(self.X, metric='euclidean')
        K = distance.squareform(K)
        if isinstance(self.epsilon, str):
            if self.epsilon == 'median':
                self.epsilon = np.median(K)
        K = np.power(K, 2)/(2*np.power(self.epsilon, 2))
        K = np.exp(-K)
        # K[K < self.sigma] = 0
        D = K.sum(axis=1)
        D = 1/D
        # D = np.sqrt(D)
        D = np.diag(D)
        P = D@K
        self.P = P

        eigenValues, eigenVectors = np.linalg.eig(P)
        eigenValues = np.real(eigenValues)
        eigenVectors = np.real(eigenVectors)
        idx = eigenValues.argsort()[::-1]
        self.eigenValues = eigenValues[idx]
        self.eigenVectors = eigenVectors[:, idx]

    def get_map(self, dim, t=None):
        if dim > self.eigenValues.shape[0]:
            print("Error: dim is too high")
            return
        vecs = self.eigenVectors[:, 1:dim+1]
        vals = self.eigenValues[1:dim+1]
        if t is None:
            self.x_map = np.tile(np.power(vals[:], self.t), (vecs.shape[0], 1))*vecs
            return self.x_map
        self.x_map = np.tile(np.power(vals[:], t), (dim, 1)) * vecs
        return self.x_map

    def nystrom_out_of_sample(self,oos_x):
        k_oos = distance.cdist(oos_x, self.X, metric='euclidean')
        k_oos = np.power(k_oos, 2)/(2*np.power(self.epsilon, 2))
        k_oos = np.exp(-k_oos)
        k_oos = k_oos/np.sum(k_oos, axis=1).reshape((k_oos.shape[0], 1))
        return k_oos@self.x_map@np.diag(1/self.eigenValues[1:1+self.x_map.shape[1]])



