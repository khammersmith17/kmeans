import numpy as np
from numpy.typing import NDArray


class KMeans:
    def __init__(self, k: int, tolerance: float = 0.001, max_iters: int = 500):
        self.k = k
        self.tolerance = tolerance
        self.max_iters = max_iters
        self._centroids = {}
        self._classes = {}

    def _euclidean_distance(self, p1: NDArray, p2: NDArray) -> float:
        # 2 norm is euclidean distance
        return np.linalg.norm(p1 - p2, axis=0)

    def predict(self, data):
        if self.centroids is None:
            raise NotImplementedError("Model needs to be fit before it is scored on")
        distances = [
            np.linalg.norm(data - self._centroids[centroid])
            for centroid in self._centroids
        ]
        return distances.index(np.min(distances))

    def fit(self, data):
        self.centroids = {}
        for i in range(self.k):
            # 'randomly' assign each cluster to a point
            self.centroids[i] = data[i]

        for i in range(self.max_iters):
            for j in range(self.k):
                self._classes[j] = []
            for point in data:
                distances = []
                for index in self.centroids:
                    # compute the eucliden distance to each cluster
                    distances.append(
                        self._euclidean_distance(point, self.centroids[index])
                    )
                # get the cluster index getting the index of the min distance
                cluster_index = distances.index(min(distances))
                self._classes[cluster_index].append(point)

            # store the previous centers to check for convergence later
            previous = dict(self._centroids)
            for cluster_index in self._classes:
                # for each cluster index take the average of each point in the cluster
                # this is to recompute the center position relative to the points in the cluster
                self.centroids[cluster_index] = np.average(
                    self._classes[cluster_index], axis=0
                )

            for centroid in self._centroids:
                # taking the distance betweent the current centroid position and the current
                # we do this to determine if we have convergence
                if (
                    np.sum(
                        (self.centroids[centroid] - previous[centroid])
                        / previous[centroid]
                        * 100.0
                    )
                    > self.tolerance
                ):
                    # if the tolerance is exceeded, we have not converged and need more iterations
                    continue
            # if we make it through the evaluation loop, there has been convergence
            return
