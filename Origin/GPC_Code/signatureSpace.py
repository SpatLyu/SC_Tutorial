import numpy as np

def signature_vector_difference(vec, relative=True):
    if not isinstance(vec, (list, np.ndarray)):
        raise ValueError("Input must be a numeric vector.")
    
    vec = np.asarray(vec, dtype=np.float64)
    
    if relative:
        # (new - old) / old
        diffs = np.diff(vec) / vec[:-1]
    else:
        # new - old
        diffs = np.diff(vec)
    
    return diffs

def signaturespace(M, relative=True):
    if not isinstance(M, np.ndarray):
        raise ValueError("Input must be a NumPy matrix or array.")
    
    if M.ndim != 2:
        raise ValueError("Input must be a 2D matrix.")
    
    E = M.shape[1]
    if E < 2:
        raise ValueError("State space matrix must have at least 2 columns.")
    
    if E == 2:
        SM = np.apply_along_axis(signature_vector_difference, 1, M, relative=relative).reshape(-1, 1)
    else:
        SM = np.apply_along_axis(signature_vector_difference, 1, M, relative=relative)
    
    SM = np.where(np.isnan(SM), np.nan, SM.astype(np.float64))  # Make sure dtype and NA are handled
    return SM
