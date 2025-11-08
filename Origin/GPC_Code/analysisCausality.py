import numpy as np
import math
from scipy.special import erf

def norm_vec(x):
    return np.sqrt(np.nansum(x ** 2))

def update_pc_heatmap_and_types(
    heatmap_accum, count_matrix, pattern_types_matrix,
    pat_x, pred_pat_y, real_pat_y,
    sig_x, pred_sig_y, real_sig_y,
    pattern_indices,
    weighted=True):

    if pat_x not in pattern_indices or pred_pat_y not in pattern_indices or real_pat_y not in pattern_indices:
        return None

    i = pattern_indices[pat_x]
    j = pattern_indices[pred_pat_y]  

    if np.isnan(pred_pat_y) or np.isnan(real_pat_y):
        return None

    if pred_pat_y == real_pat_y:
        epsilon = 1e-6
        norm_sig_x = norm_vec(sig_x) + epsilon
        strength = erf(norm_vec(pred_sig_y) / norm_sig_x) if weighted else 1
    else:
        strength = 0

    if np.isnan(heatmap_accum[i, j]):
        heatmap_accum[i, j] = strength
        count_matrix[i, j] = 1
    else:
        heatmap_accum[i, j] += strength
        count_matrix[i, j] += 1

    return (i, j, strength)

def classify_causality_type(i, j, strength, hashed_num):
    midpoint = (hashed_num - 1) / 2
    if i == j and i != midpoint:
        return "positive", strength
    elif (i + j) == (hashed_num - 1) and i != midpoint:
        return "negative", strength
    else:
        return "dark", strength

def analyze_pc_causality(
    psMx, psMy, sMx, sMy,
    pred_patY, pred_sigY,
    weighted=True):

    n = len(psMx)
    hashed_patterns = np.unique(np.concatenate([psMx, psMy, pred_patY]))
    pattern_indices = {p: idx for idx, p in enumerate(hashed_patterns)}
    hashed_num = len(hashed_patterns)

    heatmap_accum = np.full((hashed_num, hashed_num), np.nan)
    count_matrix = np.zeros((hashed_num, hashed_num))


    no_causality = np.zeros(n)
    positive = np.zeros(n)
    negative = np.zeros(n)
    dark = np.zeros(n)
    pattern_types = []
    real_loop = []

    for t in range(n):
        try:
            pat_x = psMx[t].item() if isinstance(psMx[t], np.ndarray) else psMx[t]
            pat_y_pred = pred_patY[t].item() if isinstance(pred_patY[t], np.ndarray) else pred_patY[t]
            pat_y_real = psMy[t].item() if isinstance(psMy[t], np.ndarray) else psMy[t]
        except:
            continue

        if any(np.isnan([pat_x, pat_y_real, pat_y_pred])):
            continue

        real_loop.append(t)

        res = update_pc_heatmap_and_types(
            heatmap_accum, count_matrix, pattern_types,
            pat_x, pat_y_pred, pat_y_real,
            sMx[t], pred_sigY[t], sMy[t],
            pattern_indices, weighted=weighted
        )

        if res is None:
            no_causality[t] = 1
            pattern_types.append("no_causality")
            continue

        i, j, strength = res
        if strength == 0:
            no_causality[t] = 1
            pattern_types.append("no_causality")
            continue

        ctype, s = classify_causality_type(i, j, strength, hashed_num)
        if ctype == "positive":
            positive[t] = s
        elif ctype == "negative":
            negative[t] = s
        elif ctype == "dark":
            dark[t] = s
        pattern_types.append(ctype)


    with np.errstate(invalid='ignore', divide='ignore'):
        heatmap = heatmap_accum / count_matrix


    idx = np.arange(hashed_num)
    diag_vals = heatmap[idx, idx]
    antidiag_vals = heatmap[idx, hashed_num - 1 - idx]
    other_mask = ~np.eye(hashed_num, dtype=bool) & ~(np.fliplr(np.eye(hashed_num, dtype=bool)))
    other_vals = heatmap[other_mask]

    summary = {
        "positive": np.nanmean(diag_vals),
        "negative": np.nanmean(antidiag_vals),
        "dark": np.nanmean(other_vals),
        "heatmap": heatmap
    }

    return {
        "noCausality": no_causality,
        "Positive": positive,
        "Negative": negative,
        "Dark": dark,
        "real_loop": real_loop,
        "pattern_types": pattern_types,
        "summary": summary
    }
