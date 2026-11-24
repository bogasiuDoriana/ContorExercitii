import numpy as np


def angle(a, b, c):
    """Unghiul din punctul b, in grade (0-180), format de punctele a-b-c."""
    a, b, c = (np.asarray(p, dtype=float) for p in (a, b, c))
    u, v = a - b, c - b
    norms = np.linalg.norm(u) * np.linalg.norm(v)
    if norms == 0:  # doua puncte coincid, unghiul nu este definit
        return float("nan")
    # clip: erorile de rotunjire pot scoate cosinusul din [-1, 1]
    cos = np.clip(np.dot(u, v) / norms, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))
