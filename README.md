# Intelligent Recommender System: Dimensionality Reduction & Hybrid Pipelines

This project implements a complete recommendation ecosystem designed to handle large-scale user-item interactions and the "cold-start" problem. It transitions from foundational matrix factorization techniques to a production-ready switching hybrid system.

## 🛠 Project Structure

The project is divided into two core analytical sections:

### Section 1: Dimensionality Reduction & Matrix Factorization

Focuses on mathematical approaches to decompose the user-item matrix and predict ratings:

* **PCA with Mean-Filling**: Replaces missing values with item averages to compute covariance and predict ratings via top-k peers.
* **PCA with MLE**: A more advanced approach that centers items without explicit filling, focusing only on available interactions.
* **Singular Value Decomposition (SVD)**: Implements  to find latent factors and reconstruct the rating matrix for high-accuracy collaborative filtering.

### Section 2: Hybrid Recommendation Pipeline

A multi-modal system that combines collaborative signals with item metadata:

* **Content-Based Filtering**: Uses a manual TF-IDF algorithm on text features (titles, tags, genres) and normalizes engagement metrics (views, likes) to build user profiles.
* **Item-based CF**: Implements Adjusted Cosine Similarity with a discount factor to penalize low-confidence item pairs.
* **Switching Hybrid**: A logic layer that routes "warm" users to SVD/Collaborative models and "cold" users to Content-Based/Popularity models.

## 📊 Dataset Specifications

The system is tested against two primary datasets:

1. **Movie Ratings (`ratings.csv`)**: 109,342 ratings from 14,638 users across 900 movies.
2. **Short-Form Video Data**:
* `video.csv`: Metadata including `text_features`, `genre`, and engagement metrics.
* `synthetic_interactions.csv`: Implicit signals such as `watch_ratio` and `liked` status.



## 🚀 Key Features

* **Cold-Start Mitigation**: Automatically falls back to a popularity-based mean vector for users with no history.
* **Dual-Metric Evaluation**: Measures performance using both **Exact Precision@10** (specific item hits) and **Category Precision@10** (genre relevance).
* **Interactive Dashboard**: A Streamlit-based UI (`main.py`) allowing real-time adjustment of latent factors () and hybrid switching thresholds.

## 💻 Technical Implementation

### Core Modules

* **`collaborative.py`**: Contains the logic for SVD and Item-based Collaborative Filtering.
* **`content_based.py`**: Handles NLP preprocessing, TF-IDF calculation, and user profile construction.
* **`hybrid.py`**: The core "Switching" logic that integrates all models.
* **`utils.py`**: Shared mathematical utilities for covariance and matrix filling.

### Performance & Profiling

The project includes extensive profiling for:

* **Runtime**: Comparing prediction speeds for different  neighbors.
* **Memory Usage**: Monitoring the footprint of large-scale covariance matrices and SVD embeddings.

## 📈 Evaluation Results

The `evaluation.py` script compares four distinct approaches:

1. **Hybrid (Switching)**: Best balance of accuracy and coverage.
2. **Pure Content-Based**: Reliable for new items/users.
3. **Most Popular**: Baseline for baseline engagement.
4. **Random**: Lower-bound performance baseline.