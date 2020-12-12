from abc import ABC, abstractmethod
from typing import List

import cv2
import numpy as np
from numpy import dot
from numpy.linalg import norm
from scipy.spatial import distance

from pyimg.models.image import ImageImpl


def sift_method(a_img: ImageImpl) -> (ImageImpl, List, List):
    gray_image = a_img.to_gray()

    sift = cv2.xfeatures2d.SIFT_create()
    key_points, descriptors = sift.detectAndCompute(gray_image.get_array(), None)

    return gray_image, key_points, descriptors


def compare_images_sift(
    img1: ImageImpl, img2: ImageImpl, threshold, acceptance
) -> (ImageImpl, bool, int, int, int):
    gray1, key_points1, descriptors1 = sift_method(img1)
    gray2, key_points2, descriptors2 = sift_method(img2)

    # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    bf = ChebyshevDistance()

    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    matches_qty = (np.array([x.distance for x in matches]) <= threshold).sum()

    matching_image = cv2.drawMatches(
        img1.get_array(),
        key_points1,
        img2.get_array(),
        key_points2,
        matches[:matches_qty],
        img2.get_array(),
        flags=2,
    )

    min_dimension = min(len(descriptors1), len(descriptors2))
    matching_percentage = matches_qty / min_dimension

    return (
        ImageImpl.from_array(matching_image),
        matching_percentage >= acceptance,
        len(descriptors1),
        len(descriptors2),
        matches_qty,
    )


class Matcher(ABC):
    @abstractmethod
    def match(self, query_descriptors, train_descriptors) -> List:
        """
        @brief Finds the best match for each descriptor in query_descriptors.
        """
        pass


class Match:
    """
    Class to represent a match between to descriptors.
    """

    def __init__(self, img_idx: int, query_idx: int, train_idx: int, distance: float):
        self.img_idx = img_idx
        self.query_idx = query_idx
        self.train_idx = train_idx
        self.distance = distance


class MatchAdapter:
    """
    Adapter to use the cv2 library.
    """

    @staticmethod
    def adapt(match: Match) -> cv2.DMatch:
        return cv2.DMatch(
            match.query_idx, match.train_idx, match.img_idx, match.distance
        )

    @staticmethod
    def adapt_all(matches: List) -> List:
        return [MatchAdapter.adapt(x) for x in matches]


class L2Matcher(Matcher):
    def match(self, query, train):
        query_expand = np.array([query] * train.shape[0])
        train_expand = np.array([train] * query.shape[0]).transpose(1, 0, 2)

        matches_2d = np.linalg.norm(query_expand - train_expand, axis=2)
        matches_idx = matches_2d.argmin(axis=0)
        matches_dist = matches_2d.min(axis=0)

        return MatchAdapter.adapt_all(
            [
                Match(0, i, idx, dist)
                for i, (idx, dist) in enumerate(zip(matches_idx, matches_dist))
            ]
        )


# TODO: revisar
class CosineMatcher(Matcher):
    def match(self, query, train):
        matches = []
        for i, d1 in enumerate(query):
            match = np.array([dot(d1, d2.T) / (norm(d1) * norm(d2)) for d2 in train])
            min_idx = match.argmin()
            matches.append(Match(0, i, min_idx, match[min_idx]))

        # return MatchAdapter.adapt_all([Match(0, i, idx, dist)
        #                                for i, (idx, dist) in enumerate(zip(matches_idx, matches_dist))])
        return MatchAdapter.adapt_all(matches)


class ManhattanDistance(Matcher):
    def match(self, query, train):
        query_expand = np.array([query] * train.shape[0])
        train_expand = np.array([train] * query.shape[0]).transpose(1, 0, 2)

        matches_2d = np.linalg.norm(query_expand - train_expand, axis=2, ord=1)
        matches_idx = matches_2d.argmin(axis=0)
        matches_dist = matches_2d.min(axis=0)

        return MatchAdapter.adapt_all(
            [
                Match(0, i, idx, dist)
                for i, (idx, dist) in enumerate(zip(matches_idx, matches_dist))
            ]
        )


class MinkowskiDistance(Matcher):
    def match(self, query, train):
        matches = []
        for i, d1 in enumerate(query):
            match = np.array([distance.minkowski(d1, d2) for d2 in train])
            min_idx = match.argmin()
            matches.append(Match(0, i, min_idx, match[min_idx]))

        # return MatchAdapter.adapt_all([Match(0, i, idx, dist)
        #                                for i, (idx, dist) in enumerate(zip(matches_idx, matches_dist))])
        return MatchAdapter.adapt_all(matches)


class ChebyshevDistance(Matcher):
    def match(self, query, train):
        query_expand = np.array([query] * train.shape[0])
        train_expand = np.array([train] * query.shape[0]).transpose(1, 0, 2)

        matches_2d = np.linalg.norm(query_expand - train_expand, axis=2, ord=np.inf)
        matches_idx = matches_2d.argmin(axis=0)
        matches_dist = matches_2d.min(axis=0)

        return MatchAdapter.adapt_all(
            [
                Match(0, i, idx, dist)
                for i, (idx, dist) in enumerate(zip(matches_idx, matches_dist))
            ]
        )


"""

Primera version, con un for:

for i, d1 in enumerate(query):
    match = np.array([np.linalg.norm(d1 - d2) for d2 in train])
    min_idx = match.argmin()
    matches.append(Match(0, i, min_idx, match[min_idx]))


=============================

Segunda version usando operaciones para arrays:

# query (n_descriptors1, size:128)
# train (n_descriptors2, size:128)

# query_expand - train_expand
# query_expand = np.array([query] * train.shape[0])
# query_expand (n_descriptors2, n_descriptors1, size:128)

   
query_expand:  
        0        1               nd1
0    desc1_1  desc1_2  .  .  .  (128)
1    desc1_1  desc1_2
2    desc1_1  desc1_2
3    desc1_1  desc1_2
        .        .
        .        .
        .        .
      (128)    (128)
nd2


# train_expand = np.array([train] * query.shape[0]).transpose(1,0,2)
# train_expand (n_descriptors1, n_descriptors2, size:128)
  
train_expand:  
        0        1               nd1
0    desc2_1  desc2_1  .  .  .  (128)
1    desc2_2  desc2_2
2    desc2_3  desc2_3
3    desc2_4  desc2_4
        .        .
        .        .
        .        .
      (128)    (128)
nd2

Hago la diferencia todos con todos
==> query_expand - train_expand :
              0                    1                     nd1
0    (desc1_1 - desc2_1)  (desc1_2 - desc2_1)  .  .  .  (128)
1    (desc1_1 - desc2_2)  (desc1_2 - desc2_2)
2    (desc1_1 - desc2_3)  (desc1_2 - desc2_3)
3    (desc1_1 - desc2_4)  (desc1_2 - desc2_4)
        .        .
        .        .
        .        .
      (128)    (128)
nd2

norm(query_expand - train_expand, axis=2) es de shape (n_descriptors1, n_descriptors2)
"""
