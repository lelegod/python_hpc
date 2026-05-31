import sys
import cupy as cp


def distance_matrix_oneloop(p1, p2):
    p1 = cp.radians(p1)
    p2 = cp.radians(p2)

    D = cp.empty((len(p1), len(p2)))
    for i in range(len(p1)):
        dsin2 = cp.sin(0.5 * (p1[i] - p2)) ** 2
        cosprod = cp.cos(p1[i, 0]) * cp.cos(p2[:, 0])
        a = dsin2[:, 0] + cosprod * dsin2[:, 1]
        D[i, :] = 2 * cp.arctan2(cp.sqrt(a), cp.sqrt(1 - a))

    D *= 6371
    return D


def distance_matrix_noloop(p1, p2):
    p1 = cp.radians(p1)
    p2 = cp.radians(p2)

    dsin2 = cp.sin(0.5 * (p1[:, None, :] - p2[None, :, :])) ** 2
    cosprod = cp.cos(p1[:, None, 0]) * cp.cos(p2[None, :, 0])
    a = dsin2[:, :, 0] + cosprod * dsin2[:, :, 1]
    D = 2 * cp.arctan2(cp.sqrt(a), cp.sqrt(1 - a))

    D *= 6371
    return D


def load_points(fname):
    data = cp.loadtxt(fname, delimiter=',', skiprows=1, usecols=(1, 2))
    return data


def distance_stats(D):
    assert D.shape[0] == D.shape[1], 'D must be square'
    idx = cp.triu_indices(D.shape[0], k=1)
    distances = D[idx]
    return {
        'mean': float(distances.mean()),
        'std': float(distances.std()),
        'max': float(distances.max()),
        'min': float(distances.min()),
    }


fname = sys.argv[1]
points = load_points(fname)
D = distance_matrix_noloop(points, points)
stats = distance_stats(D)
print(stats)
