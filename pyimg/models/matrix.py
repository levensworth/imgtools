import numpy as np
from scipy import ndimage


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
        return self.apply_to(other, lambda x, y: x * y)

    def mul_scalar(self, scalar: float):
        return self.apply(lambda x: x * scalar)

    def add_scalar(self, scalar: float):
        return self.apply(lambda x: x + scalar)

    def max_value(self) -> float:
        return np.amax(self.array)

    def min_value(self) -> float:
        return np.amin(self.array)

    def convolution(self, kernel_size: int, fn) -> None:
        """
        A discrete convolution implementation. Apply a function given a square window
        size of the original matrix.
        Obs:
        In this implementation we use zero padding strategy.

        :param kernel_size: int representing the size of the square sliding window
        :param fn: function to apply for each window, it should accept (np.ndarray of shape (window, window, ..)
        :return: None
        """
        # calcualte padding for each dimension except the channel's dim
        padding = self._calculate_padding(kernel_size)

        padded_matrix = self._apply_padding(padding)
        for channel in range(padded_matrix.shape[-1]):
            for index, _ in np.ndenumerate(self.array[:, :, channel]):
                # case 3d image
                i, j = index
                val = fn(
                    self._get_window(
                        padded_matrix[:, :, channel],
                        i + kernel_size,
                        j + kernel_size,
                        kernel_size,
                    )
                )
                self.array[i, j, channel] = val

    def _get_window(
        self, matrix, height: int, width: int, kernel_size: int
    ) -> np.ndarray:
        """
        given a point in the matrix, return a square window of kernel size with center in
        (height, width) position.
        :param matrix: np.ndarray
        :param height: int
        :param width: int
        :param kernel_size: int
        :return: np.ndarray
        """

        pad = int(kernel_size / 2)
        return matrix[
            int(height - pad) : int(height + pad + 1),
            int(width - pad) : int(width + pad + 1),
        ]

    def _calculate_padding(self, kernel_size: int) -> int:
        for s in self.array.shape[:-1]:
            # sanity check, skip last dimension cause it's the number of channels
            if s < kernel_size:
                raise ArithmeticError("Window is bigger that matrix!")

        return [(int(kernel_size),)]

    def _apply_padding(self, padding: [...]):
        """
        Given a 2 or 3 dim tensor apply zero padding channel wise.
        :param padding: list of paddings. eg: [(3, )] means 3 slots in each side
        :return: padded matrix
        """

        matrix = self.array
        if len(matrix.shape) == 3:

            new_dim = [int(i + 2 * padding[0][0]) for i in matrix.shape[:-1]] + [
                self.array.shape[-1]
            ]
            new_matrix = np.zeros(new_dim)

            for dim in range(matrix.shape[-1]):
                # by default 'constant' means fill with zeros
                new_matrix[:, :, dim] = np.pad(matrix[:, :, dim], padding, "constant")

        elif len(matrix.shape) == 2:
            # by default 'constant' means fill with zeros
            new_matrix = np.pad(matrix[:, :], padding, "constant")

        return new_matrix

    def convolution_fast(self, kernel: np.ndarray) -> None:
        """
        Given a kernel apply 2d convolution using zero padding to treat image edges.

        :param kernel: np.ndarray of shape (kernel_size, kernel_size, channels)
        :return: None
        """

        for dim in range(self.array.shape[-1]):
            convolution = ndimage.convolve(
                self.array[:, :, dim], kernel, mode="constant", cval=0.0
            )
            self.array[:, :, dim] = convolution

    def apply_laplacian_change(self, threshold: int, value: float):
        """
        Search vertically and horizontally for sign changes
        :param threshold: this is the minimum absolute change to be acknowledge
        :param value: value to put in places where the threshold is surpass
        """

        borders_matrix = np.zeros(self.array.shape)
        for channel in range(self.array.shape[2]):
            for i in range(self.array.shape[0]):
                for j in range(self.array.shape[1] - 1):
                    magnitude = abs(
                        self.array[i, j, channel] - self.array[i, j + 1, channel]
                    )
                    sign = self.array[i, j, channel] * self.array[i, j + 1, channel]
                    if sign >= 0 or magnitude < threshold:
                        # if we are here, it means there is no change in sign
                        borders_matrix[i, j, channel] = 0
                    else:
                        borders_matrix[i, j, channel] = value

        self.array = borders_matrix

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
