# hybrid.py
from __future__ import annotations

import pandas as pd

# --- Collaborative (Truncated SVD) ---
from collaborative import (
    load_interactions,
    build_rating_matrix,
    fill_missing_with_item_mean,
    full_svd,
    recommend_top_n_svd,
)

# --- Content-based ---
from content_based import (
    advanced_preprocessing,
    calculate_manual_tfidf,
    process_interaction_metrics,
    build_user_profiles,
    get_cold_start_profile,
    generate_recommendations,
)


def build_content_based_artifacts(videos_df: pd.DataFrame, interactions_df: pd.DataFrame):
    """
    Build the exact inputs needed for generate_recommendations():
    - item_matrix
    - user_profiles
    - cold_start_vec
    """
    # Ensure video_id exists (sometimes it's row_id)
    if "video_id" not in videos_df.columns and "row_id" in videos_df.columns:
        videos_df = videos_df.rename(columns={"row_id": "video_id"})

    if "video_id" not in videos_df.columns:
        raise ValueError("videos_df must contain 'video_id' (or 'row_id').")

    if "text_features" not in videos_df.columns:
        raise ValueError("videos_df must contain 'text_features' for content-based pipeline.")

    videos_df = videos_df.copy()
    videos_df["video_id"] = videos_df["video_id"].astype(str)

    # 1) preprocessing -> clean_tokens
    videos_df["clean_tokens"] = videos_df["text_features"].apply(advanced_preprocessing)

    # 2) manual TF-IDF
    df_tfidf, _, _ = calculate_manual_tfidf(videos_df)
    df_tfidf.index = videos_df["video_id"].values

    # 3) interaction metrics (views/likes/comments/shares) if present
    df_metrics = process_interaction_metrics(videos_df)
    df_metrics.index = videos_df["video_id"].values

    # 4) item matrix
    item_matrix = pd.concat([df_tfidf, df_metrics], axis=1)

    # 5) user profiles + cold-start vector
    user_profiles = build_user_profiles(interactions_df, item_matrix)
    cold_start_vec = get_cold_start_profile(item_matrix, videos_df, top_n=5)

    return item_matrix, user_profiles, cold_start_vec


def switching_hybrid(
    user_id: int,
    interactions_df: pd.DataFrame,
    videos_df: pd.DataFrame,
    threshold: int = 10,
    top_n: int = 10,
    k_latent: int = 20,
):
    """
    Option B Switching Hybrid:
    - If user has >= threshold ratings -> Truncated SVD
    - Else -> Content-based generate_recommendations()
    """
    user_count = int((interactions_df["user_id"] == user_id).sum())

    # Build content-based inputs once
    item_matrix, user_profiles, cold_start_vec = build_content_based_artifacts(videos_df, interactions_df)

    if user_count >= threshold:
        # --- SVD path ---
        R = build_rating_matrix(interactions_df)
        R_filled = fill_missing_with_item_mean(R)
        U, Sigma, V = full_svd(R_filled)

        recs = recommend_top_n_svd(
            target_user_id=user_id,
            R=R,
            R_filled=R_filled,
            U=U,
            Sigma=Sigma,
            V=V,
            k_latent=k_latent,
            topn=top_n,
            rating_min=1.0,
            rating_max=5.0,
        )

        out = pd.DataFrame(recs, columns=["video_id", "score"])
        out["method"] = "TruncatedSVD"
        out["history_count"] = user_count
        return out

    # --- Content-based path ---
    out = generate_recommendations(
        user_id=user_id,
        item_matrix=item_matrix,
        user_profiles=user_profiles,
        cold_start_vec=cold_start_vec,
        interactions_df=interactions_df,
        top_n=top_n,
    )
    out["history_count"] = user_count
    return out


def main():
    # =========================
    # VARIABLES (edit these)
    # =========================
    
    USER_ID = [22, 102 ,15 , 9000]



    # paths (edit to match your project)
    INTERACTIONS_PATH = "../results/part_1/interactions_clean.csv"
    VIDEOS_PATH = "../data/video.csv"

    # switching hybrid params
    THRESHOLD = 10   # >= 10 ratings => SVD, else content-based
    TOP_N = 10
    K_LATENT = 20

    # =========================
    # LOAD DATA
    # =========================
    interactions_df = load_interactions(INTERACTIONS_PATH)
    videos_df = pd.read_csv(VIDEOS_PATH)

    # =========================
    # RUN HYBRID
    # =========================
    for USER_ID in USER_ID:
        recs = switching_hybrid(
            user_id=USER_ID,
            interactions_df=interactions_df,
            videos_df=videos_df,
            threshold=THRESHOLD,
            top_n=TOP_N,
            k_latent=K_LATENT,
        )
        print(f"\nTarget user_id: {USER_ID}\n")

        # Ensure consistent schema every time
        recs = recs.copy()

        # If SVD branch returned "score", rename it
        if "score" in recs.columns and "similarity_score" not in recs.columns:
            recs = recs.rename(columns={"score": "similarity_score"})

        # Add method if missing
        if "method" not in recs.columns:
            # If content-based didn't add it, set it
            recs["method"] = "ContentBased"

        # Add history_count if missing
        if "history_count" not in recs.columns:
            recs["history_count"] = int((interactions_df["user_id"] == USER_ID).sum())

        # Add user_type if missing (SVD branch)
        if "user_type" not in recs.columns:
            recs["user_type"] = "Existing"

        # Enforce column order
        recs = recs[["video_id", "similarity_score", "user_type", "method", "history_count"]]

        print(recs.to_string(index=False))




if __name__ == "__main__":
    main()
