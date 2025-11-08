import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def map_causality_to_grid(pattern_types, positions, nrow, ncol):
    """
    Map the one-dimensional list of causality types back to a 2D grid.

    Parameters:
        pattern_types (List[str]): A list of causality types such as ['positive', 'negative', ...], length N
        positions (List[int]): Corresponding positions (e.g., indices in Mx)
        nrow, ncol (int): Number of rows and columns in the original raster matrix

    Returns:
        grid_map: np.ndarray of shape (nrow, ncol) showing encoded causality types
        mapping_dict: Dictionary mapping each causality type to a numeric code, useful for visualization
    """
    grid_map = np.full((nrow, ncol), np.nan)
    mapping = {'positive': 1, 'negative': 2, 'dark': 3, 'no_causality': 0}

    for pat, idx in zip(pattern_types, positions):
        row = idx // ncol
        col = idx % ncol
        grid_map[row, col] = mapping.get(pat, np.nan)

    return grid_map, mapping

def plot_causality_grid(pattern_types, positions, nrow, ncol):
    grid_map, mapping=map_causality_to_grid(pattern_types, positions, nrow, ncol)
    cmap = sns.color_palette("Set2", n_colors=len(mapping))
    labels = {v: k for k, v in mapping.items()}
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(grid_map, cmap=cmap, cbar_kws={'ticks': list(labels.keys())})
    cbar = plt.gca().collections[0].colorbar
    cbar.set_ticklabels([labels[i] for i in labels])
    plt.title("Spatial Distribution of Causality Types")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.show()
    
    

