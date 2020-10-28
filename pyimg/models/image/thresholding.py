import numpy as np

from pyimg.config import constants
from pyimg.models.image import ImageImpl


def global_thresholding(image: ImageImpl) -> (ImageImpl, list):
    img_array = image.get_array()
    t = []
    img_binary = []

    for c in range(image.channels):
        current_t = int(np.mean(img_array[:, :, c]))
        last_t = 0

        while abs(last_t - current_t) >= 0.5:
            last_t = current_t
            current_t = _calculate_global_threshold(img_array[:, :, c], last_t)

        img_binary.append(
            _array_binarize(img_array[:, :, c], current_t) * constants.MAX_PIXEL_VALUE
        )
        t.append(int(current_t))

    return ImageImpl.from_array(np.moveaxis(np.array(img_binary), 0, 2)), t


def _calculate_global_threshold(array: np.ndarray, t: int):
    array = array.ravel()
    g1 = array[np.where(array < t + 1)]
    g2 = array[np.where(array >= t + 1)]

    m1 = np.mean(g1)
    m2 = np.mean(g2)
    return (m1 + m2) / 2


def _array_binarize(array: np.ndarray, t: int):
    return np.where(array > t, 1, 0)


def otsu_thresholding(image: ImageImpl):
    hists = image.df()
    img_array = image.get_array()
    t = []
    img_binary = []

    for c in range(image.channels):
        prob = np.array(hists[c]) / len(img_array[:, :, c])
        prob_sumcum = np.array(ImageImpl._cdf_hist(hists[c])) / 255
        means = _compute_means(prob)
        global_means = _compute_global_mean(prob)

        variances = _compute_variance(global_means, means, prob_sumcum)
        threshold = _get_threshold_from_variance(variances)
        img_binary.append(
            _array_binarize(img_array[:, :, c], threshold) * constants.MAX_PIXEL_VALUE
        )
        t.append(threshold)

    return ImageImpl.from_array(np.moveaxis(np.array(img_binary), 0, 2)), t


def _compute_means(prob: np.ndarray) -> np.ndarray:
    means = np.zeros(constants.PIXEL_RANGE)
    for t in range(0, constants.PIXEL_RANGE):
        means[t] = 0
        for j in range(0, t + 1):
            means[t] += j * prob[j]
    return means


def _compute_global_mean(prob: np.ndarray) -> float:
    global_media = 0

    for i in range(0, constants.PIXEL_RANGE):
        global_media += i * prob[i]

    return global_media


def _compute_variance(
    global_mean: float, means: np.ndarray, prob_sumcum: np.ndarray
) -> np.ndarray:
    variances = np.zeros(constants.PIXEL_RANGE)
    for t in range(0, constants.PIXEL_RANGE):
        if int(prob_sumcum[t] == 0) or int(prob_sumcum[t]) == 1:
            variances[t] = 0
        else:
            numerator = pow(global_mean * prob_sumcum[t] - means[t], 2)
            denominator = prob_sumcum[t] * (1 - prob_sumcum[t])
            variances[t] = numerator / denominator
    return variances


def _get_threshold_from_variance(variances: np.ndarray) -> int:
    return int(np.argmax(variances))


def umbralization_with_two_thresholds(a_img: ImageImpl, high_threshold: float, low_threshold: float) -> ImageImpl:
    image_array = a_img.get_array()
    res = np.zeros_like(image_array)

    weak = np.int32(constants.MAX_PIXEL_VALUE / 2)
    strong = np.int32(constants.MAX_PIXEL_VALUE)

    strong_i, strong_j, strong_k = np.where(image_array >= high_threshold)

    weak_i, weak_j, weak_k = np.where((image_array <= high_threshold) & (image_array >= low_threshold))

    res[strong_i, strong_j, strong_k] = strong
    res[weak_i, weak_j, weak_k] = weak

    return ImageImpl.from_array(res)


