# 替代 distance_matrix.py 中的大型 pdist 距离计算

import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def distance_matrix_op(M, k=10, metric="euclidean", verbose=True):
    """
    高效版本：计算每个点的 K 个最近邻距离和索引（排除自身），避免内存爆炸。

    Parameters:
        M (np.ndarray): 原始特征矩阵，shape=(n_samples, n_features)
        k (int): 最近邻个数（不含自己）
        metric (str): 距离度量方式，例如 "euclidean"
        verbose (bool): 是否显示信息

    Returns:
        indices (np.ndarray): 每个点的最近邻索引，shape=(n_samples, k)
        distances (np.ndarray): 每个点的距离，shape=(n_samples, k)
    """
    
    if verbose:
        print(f"[KNN] Computing top-{k} {metric} distances for matrix of shape {M.shape}")

    # 构建 NearestNeighbors 模型（使用多线程）
    nbrs = NearestNeighbors(
        n_neighbors=k + 1,  # 包括自己
        algorithm='auto',
        metric=metric,
        n_jobs=-1
    )
    nbrs.fit(M)

    # 批处理防止内存爆炸
    batch_size = 10000
    n_samples = M.shape[0]
    all_distances = []
    all_indices = []

    for i in tqdm(range(0, n_samples, batch_size), desc="KNN 批处理"):
        batch = M[i:i + batch_size]
        distances, indices = nbrs.kneighbors(batch)
        all_distances.append(distances[:, 1:])  # 去掉自身距离
        all_indices.append(indices[:, 1:])      # 去掉自身索引

    distances = np.vstack(all_distances)
    indices = np.vstack(all_indices)

    return indices.astype(int), distances


# 保持兼容接口：用 neighbors_op 替代原 neighbors 函数

def neighbors_op(M, E, metric="euclidean", verbose=False):
    """
    替代原 neighbors 函数，包装 distance_matrix_op 接口。

    Parameters:
        M (np.ndarray): 输入矩阵
        E (int): 嵌入维度（返回 E 个邻居）
        metric (str): 距离度量方式
        verbose (bool): 是否输出信息

    Returns:
        indices (np.ndarray): 每个样本最近邻索引
        distances (np.ndarray): 对应距离
    """
    return distance_matrix_op(M, k=E + 1, metric=metric, verbose=verbose)
