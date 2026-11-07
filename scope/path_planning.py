"""Shortest routes over observed-free map cells, with the robot footprint applied.

A cell is traversable only when every top-down cell a footprint disk centred
anywhere inside it could touch was observed free, so each point of a planned
route also passes the per-sample clearance check that GO repeats. Unknown and
occupied cells are both treated as blocked.
"""
import math

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .mapping import FREE, OCCUPIED

SLAB_MIN_Z, SLAB_MAX_Z = .25, 1.55
SAMPLE_M = .10


class RoutePlanner:
    def __init__(self, mapping, radius_m):
        cfg = mapping.config
        self.origin = np.asarray(cfg.origin[:2], float)
        self.voxel = float(cfg.voxel_m)
        self.states = mapping.topdown(min_z=SLAB_MIN_Z, max_z=SLAB_MAX_Z)
        free = self.states == FREE
        n = int(math.ceil(radius_m / self.voxel)) + 1
        d = np.arange(-n, n + 1)
        di, dj = np.meshgrid(d, d, indexing="ij")
        # Smallest distance between any point of the centre cell and of cell (di, dj).
        gap = np.hypot(np.maximum(np.abs(di) - 1, 0), np.maximum(np.abs(dj) - 1, 0)) * self.voxel
        self.clear = (ndimage.binary_erosion(free, structure=gap <= radius_m, border_value=0)
                      if free.size else free)
        self._start = self._predecessors = self._distances = None

    def cell(self, xy):
        i, j = np.floor((np.asarray(xy, float) - self.origin) / self.voxel).astype(int)
        return int(i), int(j)

    def _inside(self, ij):
        return 0 <= ij[0] < self.clear.shape[0] and 0 <= ij[1] < self.clear.shape[1]

    def is_clear(self, xy):
        ij = self.cell(xy)
        return self._inside(ij) and bool(self.clear[ij])

    def _samples(self, a, b, step):
        a, b = np.asarray(a, float), np.asarray(b, float)
        count = max(2, int(math.ceil(np.linalg.norm(b - a) / step)) + 1)
        return np.linspace(a, b, count)

    def segment_clear(self, a, b):
        return all(self.is_clear(p) for p in self._samples(a, b, self.voxel / 4))

    def could_be_free(self, xy):
        """Inside the map and not observed occupied."""
        ij = self.cell(xy)
        return self._inside(ij) and self.states[ij] != OCCUPIED

    def line_of_sight(self, a, b, box=None):
        """No observed obstacle, and not the xy box, between a and b.

        Unknown cells do not block sight; they are what a viewpoint is for.
        """
        for p in self._samples(a, b, self.voxel / 2):
            if box is not None and np.all(p >= box[0]) and np.all(p <= box[1]):
                return False
            ij = self.cell(p)
            if self._inside(ij) and self.states[ij] == OCCUPIED:
                return False
        return True

    def reachable(self, stride=2):
        """Centres of traversable cells reachable from the start, with path cost."""
        if self._start is None:
            return np.empty((0, 2)), np.empty(0)
        cost = self._distances.reshape(self.clear.shape)[::stride, ::stride]
        i, j = np.nonzero(np.isfinite(cost))
        centres = self.origin + (np.column_stack((i, j)) * stride + .5) * self.voxel
        return centres, cost[i, j]

    def unknown_near(self, xy, radius_m):
        ij = np.asarray(self.cell(xy))
        r = int(math.ceil(radius_m / self.voxel))
        lo, hi = np.maximum(ij - r, 0), np.minimum(ij + r + 1, self.states.shape)
        if np.any(lo >= hi):
            return True
        return bool(np.any(self.states[lo[0]:hi[0], lo[1]:hi[1]] != FREE))

    def from_start(self, start_xy):
        """Single-source shortest paths over traversable cells (8-connected, no corner cutting)."""
        self._start_xy = tuple(float(v) for v in start_xy)
        if not self.is_clear(start_xy):
            self._start = None
            return False
        H, W = self.clear.shape
        index = np.arange(self.clear.size).reshape(H, W)
        rows, cols, weights = [], [], []
        for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):
            j0, j1 = max(0, -dj), W - max(0, dj)
            here = (slice(0, H - di), slice(j0, j1))
            there = (slice(di, H), slice(j0 + dj, j1 + dj))
            ok = self.clear[here] & self.clear[there]
            if di and dj:
                # Both orthogonal neighbours must be clear for a diagonal step.
                ok &= self.clear[di:H, j0:j1] & self.clear[0:H - di, j0 + dj:j1 + dj]
            ia, ib = index[here][ok], index[there][ok]
            w = math.hypot(di, dj) * self.voxel
            rows += [ia, ib]
            cols += [ib, ia]
            weights += [np.full(ia.size, w), np.full(ia.size, w)]
        graph = coo_matrix((np.concatenate(weights),
                            (np.concatenate(rows), np.concatenate(cols))),
                           shape=(self.clear.size, self.clear.size)).tocsr()
        self._start = index[self.cell(start_xy)]
        self._distances, self._predecessors = dijkstra(
            graph, indices=self._start, return_predecessors=True)
        return True

    def path_to(self, goal_xy):
        """Smoothed polyline from the start point to goal_xy, or None."""
        if self._start is None or not self.is_clear(goal_xy):
            return None
        ij = self.cell(goal_xy)
        node = ij[0] * self.clear.shape[1] + ij[1]
        if not np.isfinite(self._distances[node]):
            return None
        cells = []
        while node >= 0 and node != self._start:
            cells.append(node)
            node = self._predecessors[node]
        cells.reverse()
        ny = self.clear.shape[1]
        centres = [tuple(self.origin + (np.array([c // ny, c % ny]) + .5) * self.voxel)
                   for c in cells[:-1]]
        points = [self._start_xy, *centres, tuple(float(v) for v in goal_xy)]
        return self._smooth(points)

    def _smooth(self, points):
        result, i = [points[0]], 0
        while i < len(points) - 1:
            j = len(points) - 1
            while j > i + 1 and not self.segment_clear(points[i], points[j]):
                j -= 1
            if not self.segment_clear(points[i], points[j]):
                return None
            result.append(points[j])
            i = j
        return result


def polyline_length(points):
    return float(sum(math.dist(a, b) for a, b in zip(points, points[1:])))


def truncate(points, max_m):
    """The first max_m metres of a polyline."""
    result, travelled = [points[0]], 0.
    for a, b in zip(points, points[1:]):
        step = math.dist(a, b)
        if travelled + step >= max_m:
            t = (max_m - travelled) / step if step else 0.
            result.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
            return result
        result.append(b)
        travelled += step
    return result


def densify(points, step_m=SAMPLE_M):
    """Route samples no more than step_m apart, for clearance and deviation checks."""
    samples = [tuple(float(v) for v in points[0])]
    for a, b in zip(points, points[1:]):
        count = max(1, int(math.ceil(math.dist(a, b) / step_m)))
        for k in range(1, count + 1):
            t = k / count
            samples.append((float(a[0] + t * (b[0] - a[0])), float(a[1] + t * (b[1] - a[1]))))
    return tuple(samples)
