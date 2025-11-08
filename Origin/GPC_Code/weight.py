import numpy as np

def compute_weights(distances):
    """
    Compute weights based on the negative exponential distance formula:
    w_j = exp(-d_j) / sum(exp(-d_j))
    
    Parameters:
        distances: 1D array, distances to each nearest neighbor

    Returns:
        weights: 1D array, corresponding weight for each neighbor
    """

    distances = np.array(distances)
    weights = np.exp(-distances)

 
    weight_sum = np.sum(weights)
    if weight_sum == 0:
        weights[:] = 1.0 / len(weights)  
    else:
        weights /= weight_sum

    return weights
def weights(nearest):
    weights = [compute_weights(d) for d in nearest]
    return np.array(weights)