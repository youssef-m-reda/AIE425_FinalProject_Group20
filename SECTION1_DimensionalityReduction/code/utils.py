import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import mean_absolute_error, mean_squared_error

def get_mean_filled_matrix(ratings_df, item_avg_df):
    """
    Creates a user-item matrix where missing values are replaced by item means.
   
    """
    R = ratings_df.pivot(index='userId', columns='movieId', values='rating')
    R_filled = R.copy()
    for item in R.columns:
        avg_rating = float(item_avg_df[item_avg_df['movieId'] == item]['avg_rating'])
        R_filled[item] = R[item].fillna(avg_rating)
    return R_filled.round(2)

def compute_pca_mean_filling_cov(R_filled):
    """
    Computes covariance matrix using mean-filled data.
   
    """
    # Mean-centering the items (columns)
    item_means = R_filled.mean(axis=0)
    centered_matrix = R_filled - item_means
    
    # Sample covariance: (X^T * X) / (N - 1)
    cov_matrix = (centered_matrix.T @ centered_matrix) / (len(R_filled) - 1)
    return cov_matrix, item_means

def compute_pca_mle_cov(ratings_df, item_avg_df):
    """
    Computes covariance using MLE logic: only users who rated both items are considered.
    If no users in common, covariance is 0.
   
    """
    # Mean-center the ratings based on item averages
    merged = ratings_df.merge(item_avg_df, on="movieId", how="left")
    merged["mc_rating"] = merged["rating"] - merged["avg_rating"]
    
    items = sorted(ratings_df['movieId'].unique())
    n = len(items)
    item_to_idx = {item: i for i, item in enumerate(items)}
    cov_matrix = np.zeros((n, n))
    
    # Efficiently find products for shared users
    user_groups = merged.groupby("userId")
    for _, group in user_groups:
        item_ids = group["movieId"].values
        ratings = group["mc_rating"].values
        for (i, ri), (j, rj) in combinations(zip(item_ids, ratings), 2):
            idx_i, idx_j = item_to_idx[i], item_to_idx[j]
            prod = ri * rj
            cov_matrix[idx_i, idx_j] += prod
            cov_matrix[idx_j, idx_i] += prod
            
    # For simplicity as per requirements, MLE is estimated over specified entries
    # In standard MLE, we might divide by shared count; here we follow specific logic provided
    return pd.DataFrame(cov_matrix, index=items, columns=items)

def get_top_peers(cov_matrix, target_item, k=5):
    """
    Determines top k similar items (peers) based on the covariance matrix.
   
    """
    sims = cov_matrix[target_item].sort_values(ascending=False)
    # Exclude the item itself
    peers = sims.drop(labels=[target_item]).head(k)
    return peers

def predict_rating_pca(user_id, target_item, R_filled, cov_matrix, item_means, k=5):
    """
    Predicts a missing rating using the top-k peers from PCA space.
   
    """
    peers = get_top_peers(cov_matrix, target_item, k)
    weights = peers.values
    peer_ids = peers.index
    
    # Get user's deviations for those peers
    user_ratings = R_filled.loc[user_id, peer_ids]
    peer_means = item_means[peer_ids]
    deviations = user_ratings - peer_means
    
    numerator = np.sum(weights * deviations)
    denominator = np.sum(np.abs(weights))
    
    if denominator == 0:
        return item_means[target_item]
    
    prediction = item_means[target_item] + (numerator / denominator)
    return round(prediction, 2)

def perform_full_svd(R_filled):
    """
    Computes full SVD: R = U * Sigma * Vt.
   
    """
    U, s, Vt = np.linalg.svd(R_filled, full_matrices=False)
    return U, s, Vt

def predict_rating_svd(user_idx, item_idx, U, s, Vt, k):
    """
    Predicts rating using truncated SVD (rank k approximation).
   
    """
    # Uk: (m x k), Sk: (k x k), Vtk: (k x n)
    Uk = U[:, :k]
    Sk = np.diag(s[:k])
    Vtk = Vt[:k, :]
    
    # Prediction for specific user-item pair: r_ui = Uk[u] * Sk * Vk[i]^T
    pred = Uk[user_idx, :] @ Sk @ Vtk[:, item_idx]
    return round(pred, 2)

def calculate_metrics(actual, predicted):
    """
    Calculates MAE and RMSE.
   
    """
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return round(mae, 2), round(rmse, 2)