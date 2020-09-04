from abc import abstractmethod, ABC

import numpy as np


class Generator(ABC):

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def generate(self):
        pass


class GaussianGenerator(Generator):

    def generate(self):
        return gaussian_generator(*self.args, **self.kwargs)


class RayleighGenerator(Generator):

    def generate(self):
        return rayleigh_generator(*self.args, **self.kwargs)


class ExponentialGenerator(Generator):

    def generate(self):
        return exponential_generator(*self.args, **self.kwargs)


class UniformGenerator(Generator):

    def generate(self):
        return uniform_generator(*self.args, **self.kwargs)


def gaussian_generator(mean, std, size):
    return np.random.normal(mean, std, size)


def rayleigh_generator(xi, size):
    return np.random.rayleigh(xi, size)


def exponential_generator(alpha, size):
    beta = 1 / alpha
    return np.random.exponential(beta, size)


def uniform_generator(lowest, highest, size):
    return np.random.uniform(lowest, highest, size)
