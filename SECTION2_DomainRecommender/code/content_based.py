# -*- coding: utf-8 -*-
"""
# 3. Feature Extraction and Vector Space Model

3.1. Text feature extraction
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
df = pd.read_csv('../data/video.csv')
df = df.rename(columns={"row_id": "video_id"})




# --- DOWNLOAD REQUIRED RESOURCES ---
nltk.download('stopwords')

# --- CONFIGURATION ---
STOP_WORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

def advanced_preprocessing(text):
    """
    Performs the standard NLP pipeline:
    1. Lowercase -> 2. Clean -> 3. Tokenize -> 4. Stopwords -> 5. Stemming
    """
    if not isinstance(text, str):
        return []

    # 1. Lowercase
    text = text.lower()

    # 2. Regex: Replace non-alphanumeric chars with space
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # 3. Tokenization
    tokens = text.split()

    # 4. & 5. Stop Word Removal & Stemming combined
    clean_tokens = []
    for token in tokens:
        if token not in STOP_WORDS and len(token) > 2:
            stemmed_word = STEMMER.stem(token)
            clean_tokens.append(stemmed_word)

    return clean_tokens

# --- MANUAL TF-IDF ALGORITHM (Steps 1-4) ---

def calculate_manual_tfidf(df_input):
    docs_tokens = df_input['clean_tokens'].tolist()
    N = len(docs_tokens)

    # 1. Build Vocabulary
    unique_terms = set()
    for tokens in docs_tokens:
        unique_terms.update(tokens)

    sorted_vocab = sorted(list(unique_terms))
    vocab_index = {term: idx for idx, term in enumerate(sorted_vocab)}
    V = len(sorted_vocab)

    #print(f"Processing {N} documents with {V} unique terms...")

    # Step 1: TF
    tf_matrix = np.zeros((N, V))
    for doc_idx, tokens in enumerate(docs_tokens):
        for token in tokens:
            if token in vocab_index:
                col_idx = vocab_index[token]
                tf_matrix[doc_idx, col_idx] += 1

    # Step 2: DF
    df_counts = np.count_nonzero(tf_matrix, axis=0)

    # Step 3: IDF
    idf_values = 1 + np.log(N / (df_counts + 1e-9))

    # Step 4: TF-IDF Weights
    tfidf_matrix = tf_matrix * idf_values

    df_result = pd.DataFrame(tfidf_matrix, columns=sorted_vocab)
    return df_result, idf_values, vocab_index

# --- EXECUTION ---
# 2. Apply Preprocessing

df['clean_tokens'] = df['text_features'].apply(advanced_preprocessing)

# 3. Compute TF-IDF
tfidf_df, idf_vector, vocab_map = calculate_manual_tfidf(df)

#print("TF-IDF Matrix Computed Successfully.")
#print(tfidf_df.head())

tfidf_df.to_csv('../results/Part_2/tfidf_matrix.csv', index=False) 

"""3.2. Additional features

3.3. Create item-feature matrix
"""

# --- STEP 3.2: PROCESS ONLY INTERACTION METRICS ---

def process_interaction_metrics(df_input):
    """
    Manually processes only the popularity signals.
    Min-Max Normalization for views, likes, comments, shares.
    """

    interaction_cols = ['views', 'likes', 'comments', 'shares']

    # Create a copy to work on
    df_metrics = df_input[interaction_cols].copy()

    # Manual Min-Max Scaling: (Value - Min) / (Max - Min)
    #print("Normalizing Interaction Metrics (Views, Likes, etc.)...")
    for col in interaction_cols:
        min_val = df_metrics[col].min()
        max_val = df_metrics[col].max()

        # Prevent division by zero
        if max_val - min_val == 0:
            df_metrics[col] = 0.0
        else:
            df_metrics[col] = (df_metrics[col] - min_val) / (max_val - min_val)

    return df_metrics

# ---  STEP 3.3: CREATE ITEM-FEATURE MATRIX ---

# 1. Process the interaction metrics
df_interactions = process_interaction_metrics(df)

# 2. Combine TF-IDF (Content) + Interactions (Popularity)
# We REMOVED df_categorical (platform) from this concat
item_feature_matrix = pd.concat([tfidf_df, df_interactions], axis=1)

# Set index to video_id
item_feature_matrix.index = df['video_id']

item_feature_matrix.to_csv('../results/Part_2/item_feature_matrix.csv') 

"""# 4. User Profile Construction

4.1. Build user profiles

4.2. Handle cold-start users
"""


# --- 1. LOAD INTERACTION DATA ---
# Assuming you have this file. If not, we create a dummy structure matching your description.
# Columns: user_id, video_id, watch_ratio, watch_time, liked

df_interactions = pd.read_csv('../data/synthetic_interactions.csv')


# --- STEP 4.1: BUILD USER PROFILES (WEIGHTED AVERAGE) ---

def build_user_profiles(interactions, item_matrix):
    """
    Creates a User-Feature Matrix.
    Each user is represented by the weighted average of the videos they watched.
    Weight = watch_ratio + liked
    """

    # 1. Calculate Interaction Weight
    w_ratio = 0.7
    w_like = 0.3

    interactions['weight'] = (interactions['watch_ratio'] * w_ratio) + (interactions['liked'].astype(float) * w_like)

    # 2. Merge Interactions with Item Features

    merged_df = interactions.merge(item_matrix, left_on='video_id', right_index=True)

    # 3. Apply Weights to the Feature Vectors
    feature_cols = item_matrix.columns

    # Create a weighted matrix
    weighted_features = merged_df[feature_cols].multiply(merged_df['weight'], axis=0)

    # Add user_id back so we can group
    weighted_features['user_id'] = merged_df['user_id']

    # 4. Compute Weighted Average
    # Sum of (Vectors * Weights) / Sum of Weights

    # A. Sum of weighted vectors per user
    numerator = weighted_features.groupby('user_id').sum()

    # B. Sum of weights per user (normalization factor)
    denominator = merged_df.groupby('user_id')['weight'].sum()

    # C. Divide (Handle division by zero usually not needed if weight > 0)
    user_profiles = numerator.div(denominator, axis=0)

    return user_profiles

# Execute
item_feature_matrix
user_profile_matrix = build_user_profiles(df_interactions, item_feature_matrix)



# --- STEP 4.2: HANDLE COLD-START USERS (POPULARITY FALLBACK) ---

def get_cold_start_profile(item_matrix, df_videos, top_n=5):
    """
    Creates a generic profile based on the Top N most popular videos.
    Used for users with NO interaction history.
    """
    # 1. Find Top N videos by views
    # Ensuring we use the original video dataframe for 'views'
    top_videos = df_videos.nlargest(top_n, 'views')
    top_video_ids = top_videos['video_id'].values

    # 2. Get the feature vectors for these top videos
    # Use .loc to select by index (video_id)
    popular_vectors = item_matrix.loc[top_video_ids]

    # 3. Calculate the Mean Vector (Simple Average)
    cold_start_vector = popular_vectors.mean()

    return cold_start_vector

# Execute
cold_start_profile = get_cold_start_profile(item_feature_matrix, df)

user_profile_matrix.to_csv('../results/Part_2/user_profile_matrix.csv') 

"""# 5. Similarity Computation and Recommendation

5.1. Compute similarity

5.2. Generate top-N recommendations
"""


def generate_recommendations(user_id, item_matrix, user_profiles, cold_start_vec, interactions_df, top_n=10):
    """
    Step 5: Hybrid Recommendation Function
    1. Identifies if User is Existing or New (Cold Start).
    2. Computes Cosine Similarity between User Vector and All Items.
    3. FILTERS OUT videos the user has already watched.
    4. Returns Top-N recommendations.
    """

    # --- 1. DETERMINE USER VECTOR ---
    if user_id in user_profiles.index:
        # Existing User: Use their personalized profile
        user_vector = user_profiles.loc[user_id].values
        user_type = "Existing"
    else:
        # New User: Use the 'Popularity' fallback vector
        user_vector = cold_start_vec.values
        user_type = "Cold Start"

    # --- 2. COMPUTE SIMILARITY (Vectorized Cosine Similarity) ---
    # Convert item matrix to numpy for speed
    item_vectors = item_matrix.values
    video_ids = item_matrix.index

    # A. Dot Product (Numerator)
    dot_products = item_vectors.dot(user_vector)

    # B. Magnitudes (Denominator)
    user_mag = np.linalg.norm(user_vector)
    item_mags = np.linalg.norm(item_vectors, axis=1)

    # C. Cosine Score
    # Add epsilon to prevent division by zero
    scores = dot_products / (user_mag * item_mags + 1e-9)

    # Create a Series for easy sorting: Index=VideoID, Value=Score
    score_series = pd.Series(scores, index=video_ids)

    # --- 3. FILTER ALREADY WATCHED ITEMS ---
    if user_type == "Existing":
        # Find video_ids this user has already interacted with
        watched_videos = interactions_df[interactions_df['user_id'] == user_id]['video_id'].unique()

        # Drop them from the potential candidates
        # valid_scores contains only UNWATCHED videos
        valid_scores = score_series.drop(index=watched_videos, errors='ignore')
    else:
        # Cold start users haven't watched anything yet
        valid_scores = score_series

    # --- 4. GENERATE TOP-N RANKING ---
    # Sort descending (Highest score first)
    recommendations = valid_scores.sort_values(ascending=False).head(top_n)

    # Format Output
    results_df = pd.DataFrame({
        'video_id': recommendations.index,
        'similarity_score': recommendations.values,
        'user_type': user_type # Just for debugging to see which logic was used
    })

    return results_df

# --- EXECUTION & TESTING ---

# 1. Test for an Existing User (e.g., User 1)
# We ask for Top 10 as per requirements
#print(f"--- Recommendations for User 1 (Existing) ---")
recs_user_1 = generate_recommendations(
    user_id=1,
    item_matrix=item_feature_matrix,
    user_profiles=user_profile_matrix,
    cold_start_vec=cold_start_profile,
    interactions_df=df_interactions,
    top_n=20
)
#print(recs_user_1)

# 2. Test for a Cold Start User (e.g., User 99999)
#print(f"\n--- Recommendations for User 99999 (New) ---")
recs_cold_start = generate_recommendations(
    user_id=99999,
    item_matrix=item_feature_matrix,
    user_profiles=user_profile_matrix,
    cold_start_vec=cold_start_profile,
    interactions_df=df_interactions,
    top_n=20
)
#print(recs_cold_start)

recs_user_1.to_csv('../results/Part_2/recs_user_1.csv') 
recs_cold_start.to_csv('../results/Part_2/recs_cold_start.csv') 

"""# 6. K-Nearest Neighbors (k-NN)

6.1. Implement item-based k-NN
"""


# --- 1. PRE-COMPUTE ITEM-ITEM SIMILARITY MATRIX ---
# This is heavy! We do it once. It compares every video to every other video.
# Result: A matrix where cell [A, B] tells us how similar Video A is to Video B.

def compute_item_similarity_matrix(item_matrix):
    #print("Computing Item-Item Similarity Matrix...")

    # Convert to numpy for speed
    matrix_values = item_matrix.values

    # 1. Dot Product of Matrix with its own Transpose
    # Shape: (N_videos x N_features) . (N_features x N_videos) -> (N_videos x N_videos)
    dot_product = np.dot(matrix_values, matrix_values.T)

    # 2. Magnitudes
    norms = np.linalg.norm(matrix_values, axis=1)

    # 3. Cosine Similarity = Dot / (NormA * NormB)
    # Outer product creates a matrix of all magnitude combinations
    norm_matrix = np.outer(norms, norms)

    # Add epsilon to avoid division by zero
    similarity_matrix = dot_product / (norm_matrix + 1e-9)

    # Convert to DataFrame for easy lookup by video_id
    sim_df = pd.DataFrame(similarity_matrix, index=item_matrix.index, columns=item_matrix.index)

    return sim_df

# Calculate the matrix (run this once)
item_similarity_df = compute_item_similarity_matrix(item_feature_matrix)

# --- 2. IMPLEMENT k-NN PREDICTION (Your Reference Logic) ---

def predict_rating_knn(user_id, target_video_id, item_sim_df, interactions_df, k=10):
    """
    Predicts a score (0 to 1) for a specific user and video using k-NN.
    Formula: Weighted Average of the user's ratings on Neighbor Videos.
    """

    # 1. Get the User's History
    # We need to know what they watched and their 'rating' (watch_ratio)
    user_history = interactions_df[interactions_df['user_id'] == user_id]

    if user_history.empty:
        return 0.0 # Cold start user has no history to vote with

    # Map video_id to the rating given
    # We use 'watch_ratio' as the implicit rating
    user_ratings = dict(zip(user_history['video_id'], user_history['watch_ratio']))

    # 2. Find k-Nearest Neighbors for the Target Video
    if target_video_id not in item_sim_df.index:
        return 0.0 # Unknown video

    # Get all similarities for the target video, sort descending
    # Drop the video itself (similarity 1.0)
    all_neighbors = item_sim_df[target_video_id].drop(target_video_id).sort_values(ascending=False)

    # Select Top K (e.g., 20)
    k_neighbors = all_neighbors.head(k)

    # 3. Calculate Weighted Average (The Reference Formula)
    numerator = 0.0
    denominator = 0.0

    found_any_match = False

    for neighbor_id, similarity_score in k_neighbors.items():
        # Only consider neighbors the user has actually RATED/WATCHED
        if neighbor_id in user_ratings:
            user_rating = user_ratings[neighbor_id]

            # Sum(Similarity * Rating)
            numerator += similarity_score * user_rating

            # Sum(Similarity)
            denominator += similarity_score

            found_any_match = True

    # 4. Final Calculation
    if not found_any_match or denominator == 0:
        return 0.0 # User hasn't watched any similar videos

    predicted_score = numerator / denominator
    return predicted_score

# --- EXECUTION & TEST ---

# Let's test: Will User 1 like Video 103?
# Based on whether they liked the *neighbors* of Video 103.

target_vid = '1b001a2512724360670129b01488aa50' # Assuming this ID exists in your data
user_id_test = 1

prediction = predict_rating_knn(
    user_id=user_id_test,
    target_video_id=target_vid,
    item_sim_df=item_similarity_df,
    interactions_df=df_interactions,
    k=20
)

#print(f"k-NN Prediction for User {user_id_test} on Video {target_vid}: {prediction:.4f}")
