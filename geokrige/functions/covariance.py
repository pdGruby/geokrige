import numpy as np


def covariance_gaussian_func(distance, range_param, sill_param):
    return sill_param * np.exp(-distance ** 2 / (2 * range_param ** 2))


def covariance_linear_func(distance, range_param, sill_param):
    if 0 <= np.abs(distance) <= range_param:
        return sill_param * (1 - (np.abs(distance)/range_param))
    return 0


def covariance_exponential_func(distance, range_param, sill_param):
    return sill_param * np.exp(-np.abs(distance)/range_param)


def covariance_spherical_func(distance, range_param, sill_param):
    if 0 <= np.abs(distance) <= range_param:
        return sill_param * (1 - (1.5 * np.abs(distance)/range_param) + (0.5 * (np.abs(distance)/range_param)**3))
    return 0
