# vim: expandtab:ts=4:sw=4
import numpy as np


def pairdist(a, b):
    a, b = np.asarray(a), np.asarray(b) # storing as np array
    if len(a) == 0 or len(b) == 0: # no vectors for pair wise distance calculation
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1) # square sum along the m components or row
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :] # squared distance between the m component vectors
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


def cosine_distance(a, b, data_is_normalized=False):
    # for finding the cosine distance by dot product of normalized data
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def nn_euclidean_distance(x, y):
    # finding the minimum distance for each points in y to points in x using the pair dist as helper function
    distances = pairdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))


def nn_cosine_distance(x, y):
    # same as  nn_euclidean_distance but using cosine_distance as helper function
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)


class NearestNeighborDistanceMetric(object):
    #This is used as a metric that gives  for each target the closest distance to any sample that has been observed so far.
    def __init__(self, metric, matching_threshold, budget=None):

        # function defined above
        if metric == "euclidean":
            self._metric = nn_euclidean_distance
        elif metric == "cosine":
            self._metric = nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix