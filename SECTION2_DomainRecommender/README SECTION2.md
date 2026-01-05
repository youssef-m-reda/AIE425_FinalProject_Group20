# 🎥 Short-Form Video Recommender System

This repository contains a full-stack recommendation pipeline designed for short-form video platforms. The system addresses the **Cold Start** problem and optimizes for user engagement by combining behavioral data (collaborative) with metadata signals (content-based).

## 🛠 Project Structure

* `collaborative.py`: Implements **Item-based Collaborative Filtering** (using Adjusted Cosine Similarity with a significance discount) and **Matrix Factorization** via Truncated SVD.
* `content_based.py`: Features an NLP pipeline (Stemming/Stop-word removal), manual **TF-IDF vectorization**, and user profile construction based on weighted interaction metrics (watch ratio and likes).
* `hybrid.py`: Logic for the **Switching Hybrid** model. It dynamically routes users based on their interaction history:
* **Existing Users** (History  Threshold): Powered by Truncated SVD.
* **New/Cold-Start Users** (History  Threshold): Powered by Content-Based filtering or Popularity Fallbacks.


* `evaluation.py`: Evaluates model performance using **Exact Precision@10** and **Category (Soft) Precision@10**.
* `main.py`: A **Streamlit dashboard** providing a real-time UI to test recommendations for any user ID.



## 🚀 Key Algorithms

### 1. Collaborative Filtering (Memory & Model-Based)

* **Item-CF**: Uses Adjusted Cosine Similarity to normalize user rating scales. A "beta" discount factor is applied to penalize item pairs with low common support.
* **Truncated SVD**: Decomposes the User-Item matrix into latent factors () to predict ratings for unobserved items, effectively capturing "hidden" patterns in user behavior.

### 2. Content-Based Filtering

* **Vectorization**: Converts video titles, tags, and genres into a TF-IDF matrix.
* **Interaction Weighting**: User profiles are not just averages of what they watched; they are weighted vectors where:



### 3. Switching Hybrid

To maximize accuracy, the system uses a threshold-based switch (default: 10 ratings).


## 📊 Evaluation Results

The models were evaluated on a held-out test set (20% of interactions per user). We measured **Exact Precision** (hitting the specific video) and **Category Precision** (hitting the correct genre).

| Approach | Exact Precision@10 | Category Precision@10 |
| --- | --- | --- |
| **Hybrid (Switching)** | **High** | **Very High** |
| Pure Content-Based | Medium | High |
| Most Popular | Low | Medium |
| Random | Very Low | Very Low |



## 💻 How to Run

### Prerequisites
pip install pandas numpy scikit-learn nltk streamlit


### Run the UI

To launch the interactive recommender dashboard: streamlit run main.py

### Run Evaluation

To see the benchmark comparisons:python evaluation.py
