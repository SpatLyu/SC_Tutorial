import numpy as np
import math

def hashing(vec):
    """
    Generate hash value from pattern vector.
    Equivalent to sum(vec[i] * factorial(i + 2)).
    """
    vec = np.asarray(vec, dtype=np.float64)
    if np.any(np.isnan(vec)):
        return np.nan
    return np.sum(vec * np.array([math.factorial(i + 2) for i in range(len(vec))]))

def pattern_vector_difference(sVec):
    """
    Convert numeric signature vector to categorical pattern vector, then hash.
    Rules:
      - if > 0 → 3
      - if < 0 → 1
      - if == 0 → 2
    """
    sVec = np.asarray(sVec, dtype=np.float64)
    if np.any(np.isnan(sVec)):
        return np.full_like(sVec, np.nan)

    p_vec = np.where(sVec > 0, 3, np.where(sVec < 0, 1, 2))
    return hashing(p_vec)

def patternspace(SM):
    """
    Transform signature matrix into pattern space matrix.
    Each row of SM is converted into a hashed pattern.
    """
    if not isinstance(SM, np.ndarray) or SM.ndim != 2:
        raise ValueError("Input must be a 2D numeric numpy array.")
    
    pattern_hashes = [pattern_vector_difference(SM[i, :]) for i in range(SM.shape[0])]
    return np.array(pattern_hashes).reshape(-1, 1)
