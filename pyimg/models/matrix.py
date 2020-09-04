import numpy as np


class Matrix:
    def __init__(self, array):
        self.array = np.array(array).astype(float)

    def apply(self, op):
        return self.from_array(op(self.array))

    def apply_to(self, other, op):
        return self.from_array(op(self.array, other.array))

    def add(self, other):
        return self.apply_to(other, lambda x, y: x + y)

    def sub(self, other):
        return self.apply_to(other, lambda x, y: x - y)

    def mul(self, other):
        return self.apply_to(other, lambda x, y: np.matmul(x, y))

    def mul_scalar(self, scalar: float):
        return self.apply(lambda x: x * scalar)

    def add_scalar(self, scalar: float):
        return self.apply(lambda x: x + scalar)

    def max_value(self) -> float:
        return np.amax(self.array)

    def min_value(self) -> float:
        return np.amin(self.array)

    @staticmethod
    def from_array(array):
        return Matrix(array)

    @staticmethod
    def apply_op(m1, op):
        return m1.apply(op)

    @staticmethod
    def apply_binary_op(m1, m2, op):
        return m1.apply(m2, op)

    @staticmethod
    def add_matrix(m1, m2):
        return m1.add(m2)

    @staticmethod
    def sub_matrix(m1, m2):
        return m1.sub(m2)

    @staticmethod
    def mul_matrix(m1, m2):
        return m1.mul(m2)

    @staticmethod
    def mul_scalar_matrix(m1, scalar):
        return m1.mul_scalar(scalar)

    @staticmethod
    def add_scalar_matrix(m1, scalar):
        return m1.add_scalar(scalar)

