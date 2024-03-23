import numpy as np


def variogram_gaussian_func(distance, range_param, sill_param):
    return sill_param * (1 - np.exp(-distance ** 2 / (2 * range_param ** 2)))


def variogram_linear_func(distance, range_param, sill_param):
    if 0 <= np.abs(distance) <= range_param:
        return sill_param * (np.abs(distance) / range_param)
    return sill_param


def variogram_exponential_func(distance, range_param, sill_param):
    return sill_param * (1 - np.exp(-np.abs(distance) / range_param))


def variogram_spherical_func(distance, range_param, sill_param):
    if 0 <= np.abs(distance) <= range_param:
        return sill_param * ((1.5 * np.abs(distance)/range_param) - (0.5 * (np.abs(distance)/range_param)**3))
    return sill_param
