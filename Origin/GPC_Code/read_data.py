import rasterio
import numpy as np

def read_raster(raster_path):
    """
    Read a raster file into a NumPy array, and return its data, affine transform, and metadata.

    Parameters:
        raster_path (str): Path to the raster file (supports formats such as .tif, .img, etc.)

    Returns:
        data (np.ndarray): Raster data array (shape is (bands, rows, cols) for multi-band, (rows, cols) for single-band)
        transform (Affine): Affine transformation information
        meta (dict): Original metadata of the raster
    """

    with rasterio.open(raster_path) as src:
        data = src.read()  #
        transform = src.transform
        meta = src.meta
    

    if data.shape[0] == 1:
        data = data[0]
    
    return data, transform, meta


    
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_matrix_data(matrix, title="Original Matrix", cmap="viridis", show_values=True):
    """
    Visualize the raw matrix data.

    Parameters:
        matrix (2D np.ndarray): Raw data matrix
        title (str): Plot title
        cmap (str): Colormap (e.g., 'viridis', 'coolwarm')
        show_values (bool): Whether to display values on the heatmap
    """

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(matrix, annot=show_values, fmt=".1f", cmap=cmap, cbar=True,
                     linewidths=0.5, linecolor='gray', square=True)
    plt.title(title)
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.tight_layout()
    plt.show()
