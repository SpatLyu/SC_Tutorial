import numpy as np
from signatureSpace import signaturespace
from patternSpace import patternspace
from distance import distance_matrix

def transtoM(Matrix,E,tau,metric="euclidean", as_matrix=True, verbose=False):
    embeddings=[]
    # First embedding: flatten the transpose
    embeddings.append(Matrix.T.flatten())
    for i in range(1,E+1):
        lagged = lagged_variable(Matrix, i * tau)
        lagged = np.nanmean(lagged, axis=1)
        embeddings.append(lagged)
    embeddings = np.array(embeddings)
    M = embeddings.T
    sp = signaturespace(M)
    ps = patternspace(sp)
    D=distance_matrix(M,metric, as_matrix, verbose)
    
    if verbose:
        print(f"Computing M for matrix of shape {M.shape}")
        print(f"signaturespace matrix of shape {sp.shape}")
        print(f"patternspace matrix of shape {ps.shape}")
        print(f"Distance matrix of shape {D.shape}")
              
    
    return M,sp,ps,D

def transtoM2(Matrix,E,tau,metric="euclidean", as_matrix=True, verbose=False):
    embeddings=[]
    # First embedding: flatten the transpose
    embeddings.append(Matrix.T.flatten())
    for i in range(1,E+1):
        lagged = lagged_variable(Matrix, i * tau)
        lagged = np.nanmean(lagged, axis=1)
        embeddings.append(lagged)
    embeddings = np.array(embeddings)
    M = embeddings.T
    sp = signaturespace(M)
    ps = patternspace(sp)
    #D=distance_matrix(M,metric, as_matrix, verbose)
    
    if verbose:
        print(f"Computing M for matrix of shape {M.shape}")
        print(f"signaturespace matrix of shape {sp.shape}")
        print(f"patternspace matrix of shape {ps.shape}")
        #print(f"Distance matrix of shape {D.shape}")
              
    
    return M,sp,ps#,D


def expand_matrix(data_matrix, lag_num):
    if lag_num < 0:
        return data_matrix
    
    lag_num = int(round(lag_num))
    
    if lag_num > 1:
        data_matrix = expand_matrix(data_matrix, lag_num - 1)
    
    row_num, col_num = data_matrix.shape
    
    # 上下添加 NaN 行
    top = np.full((1, col_num), np.nan)
    bottom = np.full((1, col_num), np.nan)
    data_matrix = np.vstack([top, data_matrix, bottom])
    
    # 左右添加 NaN 列
    left = np.full((data_matrix.shape[0], 1), np.nan)
    right = np.full((data_matrix.shape[0], 1), np.nan)
    data_matrix = np.hstack([left, data_matrix, right])
    
    return data_matrix

def lagged_variable(data_matrix, lag_num):
    col_num = data_matrix.shape[1]
    row_num = data_matrix.shape[0]
    ex_data_matrix = expand_matrix(data_matrix, lag_num)

    lagged_var = np.full((row_num * col_num, 8 * lag_num), np.nan)

    for r in range(row_num):
        for c in range(col_num):
            item = 0
            exr = r + lag_num
            exc = c + lag_num

            # Start from North (NE to NW)
            for la in range(lag_num, 0, -1):  # R: seq(lagNum, 1-lagNum)
                lagged_var[r * col_num + c, item] = ex_data_matrix[exr - lag_num, exc + la]
                item += 1

            # West (NW to SW)
            for ra in range(-lag_num, lag_num):  # R: seq(-lagNum, lagNum-1)
                lagged_var[r * col_num + c, item] = ex_data_matrix[exr + ra, exc - lag_num]
                item += 1

            # South (SW to SE)
            for la in range(-lag_num, lag_num):  # R: seq(-lagNum, lagNum-1)
                lagged_var[r * col_num + c, item] = ex_data_matrix[exr + lag_num, exc + la]
                item += 1

            # East (SE to NE)
            for ra in range(lag_num, 0, -1):  # R: seq(lagNum, 1-lagNum)
                lagged_var[r * col_num + c, item] = ex_data_matrix[exr + ra, exc + lag_num]
                item += 1

    return lagged_var