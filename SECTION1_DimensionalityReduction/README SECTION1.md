### Section 1: Dimensionality Reduction and Matrix Factorization

This section of the project focuses on implementing and comparing various dimensionality reduction techniques, Principal Component Analysis (PCA) and Singular Value Decomposition (SVD), for collaborative filtering in a recommender system. The goal is to predict missing user ratings for specific target items by capturing latent structures in the data.


## Project Overview

The implementation is divided into three main parts:

1. PCA with Mean-Filling: Using item averages to handle missing data before computing the covariance matrix and predicting ratings.
2. PCA with Maximum Likelihood Estimation (MLE): Estimating covariance only from specified entries (users who rated both items) to avoid bias from mean-filling.
3. Singular Value Decomposition (SVD): Decomposing the user-item matrix into latent factors to approximate the full matrix and handle cold-start scenarios.


## Dataset Requirements

The system uses a preprocessed subset of the MovieLens dataset (or similar) meeting these criteria:

- Users: > 10,000
- Items: > 500 products
- Interactions: > 100,000 ratings
Scale: Ratings adjusted to a 1–5 scale

## Implementation Details

# Part 1: PCA with Mean-Filling

- Preprocessing: Missing ratings are replaced with the item's average rating.
- Covariance Matrix: A sample covariance matrix is generated from the mean-centered ratings.
- Dimensionality Reduction: The top 5 and top 10 principal components are extracted via eigen-decomposition.
- Prediction: Missing ratings for target items () are calculated using the weighted deviations of the top -peers ().

# Part 2: PCA with MLE

- MLE Approach: Unlike Part 1, the covariance between a pair of items is estimated using only the users who have rated both items. If no common users exist, the covariance is set to 0.
- Embeddings: User embeddings are generated in the reduced dimensional space (PC1 to PC5/PC10).
- Comparison: Results are compared against Part 1 to analyze the impact of using only observed data versus mean-filled data on prediction accuracy.

# Part 3: Singular Value Decomposition (SVD)

- Full SVD: Decomposes the ratings matrix  into .
- Truncated SVD: Implements low-rank approximations for .
- Evaluation:Elbow Method: Used to identify the optimal number of latent factors by plotting reconstruction error (MAE/RMSE) against .
- Cold-Start Analysis: Performance is tested on simulated "cold" users with fewer than 5 ratings.
- Hybrid Mitigation: Combines SVD with item popularity to improve cold-start recommendations.


## Results and Visualizations
The notebooks generate several key visualizations to support the analysis:

- Scree Plots: Showing the variance explained by each singular value.
- Elbow Curves: Plotting reconstruction error vs. number of latent factors ().
- Latent Space Projections: 2D scatter plots projecting users and items onto the first two latent factors.
- Statistical Distributions: Plots showing the distribution of ratings per item and popularity groupings ( to ).

File Structure

- `SECTION1.ipynb`: Data preparation, sampling, and initial statistical analysis.
- `pca_mean_filling.ipynb`: Implementation of PCA using the mean-filling technique.
- `pca_mle.ipynb`: Implementation of PCA using Maximum Likelihood Estimation.
- `svd_analysis.ipynb`: Comprehensive SVD implementation, including truncated SVD and cold-start analysis.