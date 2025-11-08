import math
import numpy as np

def decode_pattern_hashes(pattern_hashes):
      """
    Convert pattern hash values back to arrow representations (↑↓→).

    Parameters:
    pattern_hashes: list or array containing all unique pattern hash values (e.g., 152, 168, ...)

    Returns:
    dict: {hash value: string of arrow symbols}
        """
    arrow_map = {1: "↓", 2: "→", 3: "↑"}
    decoded_patterns = {}

    for h in pattern_hashes:
        pattern = []
        rem = int(h)
        for i in reversed(range(1, 10)):  # 
            base = math.factorial(i + 1)
            digit = rem // base
            rem = rem % base
            if digit in arrow_map:
                pattern.append(arrow_map[digit])
            else:
                break
        decoded_patterns[h] = ''.join(pattern[::-1])  
    return decoded_patterns


import seaborn as sns
import matplotlib.pyplot as plt

def plot_pattern_heatmap_with_arrows(heatmap_matrix, pattern_hashes):
    """
    Replace axis labels of the Pattern Causality Heatmap with arrow representations.

    Parameters:
        heatmap_matrix: 2D matrix (P x P) representing average causal strengths
        pattern_hashes: List or array of hash values corresponding to each row and column
    """
    label_map = decode_pattern_hashes(pattern_hashes)
    arrow_labels = [label_map[h] for h in pattern_hashes]

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_matrix, cmap="coolwarm", annot=True, fmt=".2f",
                xticklabels=arrow_labels, yticklabels=arrow_labels)
    plt.title("Pattern Causality Strength Matrix (Arrows)")
    plt.xlabel("Predicted Pattern")
    plt.ylabel("Causal Pattern")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from GPC import geo_pattern_causality

def plot_total_causality_over_lags(
    xMatrix,
    yMatrix,
    E_range=range(1, 11),
    tau=1,
    lag=7,
    metric="euclidean",
    weighted=True,
    verbose=True,
    show=True
):
    """
    Plot the effect of lag order (tau) on total causal strength.

    Parameters:
        x_matrix, y_matrix: Original spatial data matrices
        E: Embedding dimension
        tau_range: List of lag orders (e.g., range(1, 11))
        lag: Spatial embedding distance
        metric: Distance metric
        weighted: Whether to apply weighting
        show: Whether to display the plot

    Returns:
        tau_list, total_strengths: List of tau values and corresponding total causal strengths
    """

    tau_list = []
    total_strengths = []
    
    for E in tqdm(E_range, desc="Computing PC Causality by tau"):
        try:
            results = geo_pattern_causality(
                xMatrix, yMatrix,
                lag=lag, E=E, tau=tau,
                metric=metric, weighted=weighted, verbose= verbose
                 
            )
            total_strength = results["summary"]["positive"] + \
                             results["summary"]["negative"] + \
                             results["summary"]["dark"]
            tau_list.append(E)
            total_strengths.append(total_strength)
        except Exception as e:
            print(f"[Tau={E}] Failed: {e}")
            continue

    if show:
        plt.figure(figsize=(8, 5))
        plt.plot(tau_list, total_strengths, marker='o', color='blue')
        plt.title("Total Pattern Causality vs E")
        plt.xlabel("Time Delay (E)")
        plt.ylabel("Total Causality Strength")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return tau_list, total_strengths
