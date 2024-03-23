import numpy as np

from geokrige.base import KrigingBase


class OrdinaryKriging(KrigingBase):
    def _predict_values(self):
        covariance_dist_matrix = self.variogram_model.covariance_func(self.dist_matrix, *self.learned_params)
        weights = self.cov_learned_inv @ covariance_dist_matrix

        predicted_values = self.y @ weights
        return predicted_values


class SimpleKriging(KrigingBase):
    def _predict_values(self):
        covariance_dist_matrix = self.variogram_model.covariance_func(self.dist_matrix, *self.learned_params)
        weights = self.cov_learned_inv @ covariance_dist_matrix

        ones = np.ones(self.y.shape[0])
        weights_mean = self.cov_learned_inv @ ones / self.cov_learned_inv.sum()
        mean_krig = self.y @ weights_mean

        predicted_values = mean_krig + (self.y - mean_krig) @ weights
        return predicted_values


class UniversalKriging(KrigingBase):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("Universal Kriging is not supported yet.")

    def _predict_values(self):
        raise NotImplementedError

    def _create_cov_matrix(self):
        raise NotImplementedError
