from pathlib import Path
import numpy as np
from trans2M import transtoM
from read_data import read_raster
from distance import neighbors
from weight import weights
from prediction import predictionY
from analysisCausality import analyze_pc_causality
from distance import neighbors


def geo_pattern_causality(
    xMatrix,
    yMatrix,
    E: int = 3,
    tau: int = 1,
    lag: int = 7,
    metric: str = "euclidean",
    weighted: bool = True,
    verbose: bool = True
):
    """
    Main wrapper function: reads raster data, constructs embedding, computes neighbors and weights,
    performs prediction and analyzes causality.

    Parameters:
        xMatrix: np.ndarray - Raster matrix for variable X
        yMatrix: np.ndarray - Raster matrix for variable Y
        E: int - Embedding dimension
        tau: int - Time delay
        lag: int - Padding boundary (used in embedding)
        metric: str - Distance metric
        weighted: bool - Whether to apply weighted prediction
        verbose: bool - Whether to output detailed logs

    Returns:
        results: dict - Contains causality type, strength, causality distribution, real_loop, etc.
    """

   # Construct embedding space
    Mx, sMx, psMx,Dx= transtoM(xMatrix, E, tau, metric, as_matrix=True, verbose=verbose)#, Dx 
    My, sMy, psMy,Dy= transtoM(yMatrix, E, tau, metric, as_matrix=True, verbose=verbose)

     # Nearest neighbors and weights
    nearest_dis_x, nearest_index_x = neighbors(Dx, E)
    weight_x = weights(nearest_dis_x)

    # Signature and pattern prediction
    pred_sigY, pred_patY = predictionY(sMy, nearest_index_x, weight_x)

    # Causality analysis
    results = analyze_pc_causality(psMx, psMy, sMx, sMy, pred_patY, pred_sigY, weighted=weighted)
    summary = results["summary"]
 
    return results

