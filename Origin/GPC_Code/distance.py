from scipy.spatial.distance import pdist, squareform
import numpy as np

def distance_matrix(M, metric="euclidean", as_matrix=True, verbose=False):
    if not isinstance(M, np.ndarray) or M.ndim != 2:
        raise ValueError("M must be a 2D numpy array.")

    if metric not in ["euclidean", "manhattan", "chebyshev"]:
        raise ValueError("Unsupported metric.")

    if verbose:
        print(f"Computing {metric} distance for matrix of shape {M.shape}")
    
    d = pdist(M, metric=metric)
    if as_matrix:
        if verbose:
            print("Converting to full matrix")
        return squareform(d)
    else:
        return d

def neighbors(D, E, valid_ratio=0.0):
    all_indices=[]
    all_dists = []
    for p in range(D.shape[0]):
        dist_vector = D[p, :]
        dist_vector[p] = np.nan 
        indices, dists = nearest_neighbors(dist_vector, E, valid_ratio=0.1)
        all_indices.append(indices)  
        all_dists.append(dists)
     

    return np.array(all_indices),np.array(all_dists)
        
    
def nearest_neighbors(distances, E, valid_ratio=0.0):
    """
    Select E+1 nearest neighbors from the distance vector, and further filter neighbors 
    based on the threshold: max + validRatio * mean.

    Parameters:
        distances (np.ndarray): 1D array of distances (e.g., from one point to all others)
        E (int): Embedding dimension
        valid_ratio (float): Expansion factor (e.g., 0.1)

    Returns:
        final_indices (np.ndarray): Filtered indices (can be used to index Mx)
        nearest_distances (np.ndarray): Distances of selected points
    """
    # Preserve original indices
    all_indices = np.arange(len(distances))
    
    # Exclude NaN values
    valid_mask = ~np.isnan(distances)
    valid_distances = distances[valid_mask]
    valid_indices = all_indices[valid_mask]

    # If too few valid points, return empty
    if len(valid_distances) <= E:
        return np.array([], dtype=int), np.array([])

    # Sort distances
    sorted_idx = np.argsort(valid_distances)
    nearest_dists = valid_distances[sorted_idx][:E+1]
    nearest_inds = valid_indices[sorted_idx][:E+1]

#    
#     mean_dist = np.nanmean(nearest_dists)
#     max_dist = np.nanmax(nearest_dists)
#     dist_threshold = max_dist + valid_ratio * mean_dist

#    
#     selected = np.where(valid_distances <= dist_threshold)[0]
#     final_indices = valid_indices[selected]
#     final_dists = valid_distances[selected]

    return nearest_dists,nearest_inds


        