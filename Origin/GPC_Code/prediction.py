import numpy as np
from patternSpace import pattern_vector_difference

def prediction_sign_Y(signatures, weights, zero_tolerance=None):
     """
    Given the signatures and weights of Y's nearest neighbors, predict its signature and the corresponding pattern.

    Parameters:
    signatures (np.ndarray): shape = (E+1, E−1), each row is a signature from one neighbor
    weights (np.ndarray): shape = (E+1,), corresponding weights for each neighbor
    zero_tolerance (int): Threshold for number of zeros allowed per dimension before setting the prediction to zero

    Returns:
    result: dict
        {
            "signature": predicted_signature_y,
            "pattern": predicted_pattern_y,
            "parameters": {
                "E": E,
                "zeroTolerance": zero_tolerance
            }
        }
    """

    if signatures.ndim == 1:
        # 1D 情况（E=2）
        signatures = signatures.reshape(-1, 1)

    E = signatures.shape[1] + 1
    if zero_tolerance is None:
        zero_tolerance = E - 1

    predicted_signature_y = np.full(signatures.shape[1], np.nan)

    for part in range(signatures.shape[1]):
        zero_count = np.sum(signatures[:, part] == 0)
        if zero_count > zero_tolerance:
            predicted_signature_y[part] = 0
        else:
            predicted_signature_y[part] = np.nansum(signatures[:, part] * weights)

    predicted_pattern_y = pattern_vector_difference(predicted_signature_y)

    return {
        "signature": predicted_signature_y,
        "pattern": predicted_pattern_y,
        "parameters": {
            "E": E,
            "zeroTolerance": zero_tolerance
        }
    }


def predictionY(sMy, nearest_index_x, weights_x, zero_tolerance=None):
    """
    Batch call of predictionY to perform prediction for all points.

    Parameters:
        sMy: np.ndarray, shape (N, E-1), signature space of Y
        nearest_index_x: list of arrays, each containing E+1 neighbor indices for a point
        weights_x: list of arrays, corresponding weights for the neighbors
        zero_tolerance: int, default is E-1, used to control tolerance in signature prediction

    Returns:
        predicted_signatures: np.ndarray, shape (N, E-1)
        predicted_patterns: np.ndarray, shape (N,)
    """

    predicted_signatures = []
    predicted_patterns = []
    nearest_index_x=nearest_index_x.astype(int)
    for i in range(len(nearest_index_x)):
        neighbors = nearest_index_x[i]
        weights = weights_x[i]

        if len(neighbors) < 1 or len(weights) != len(neighbors):
            predicted_signatures.append(np.full(sMy.shape[1], np.nan))
            predicted_patterns.append(np.nan)
            continue

        sig_neighbors = sMy[neighbors]
        result = prediction_sign_Y(sig_neighbors, weights, zero_tolerance)

        predicted_signatures.append(result["signature"])
        predicted_patterns.append(result["pattern"])

    return np.vstack(predicted_signatures), np.array(predicted_patterns).reshape(-1, 1)

