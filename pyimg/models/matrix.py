import numpy as np


class Matrix:

    def __init__(self, array):
        self.array = np.array(array).astype(float)

    def apply(self, op):
        return op(self)

    def apply_to(self, other, op):
        return op(self, other)

    def add(self, other):
        return self.apply_to(other, lambda x, y: x.array + y.array)

    def diff(self, other):
        return self.apply_to(other, lambda x, y: x.array - y.array)

    def mul(self, other):
        return self.apply_to(other, lambda x, y: np.matmul(x.array, y.array))

    def mul_scalar(self, scalar):
        return self.apply(lambda x: x.array * scalar)
