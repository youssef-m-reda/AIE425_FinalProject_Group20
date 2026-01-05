# # main.py
# # Runs: Item-based CF, Truncated SVD, Content-based, Hybrid Switching
# # Uses ONLY: collaborative.py, content_based.py, hybrid.py

# import pandas as pd
# from pathlib import Path

# # ===============================
# # Variables (edit if needed)
# # ===============================

# TOP_N = 10
# K_LATENT = 20             # truncation rank for SVD
# HYBRID_THRESHOLD = 10     # >= threshold => SVD, else content-based

# # ItemCF hyperparams (these exist in collaborative.py main())
# K_NEIGHBORS = 250
# BETA = 50
# MIN_COMMON = 2
# RATING_MIN = 1.0
# RATING_MAX = 5.0

# INTERACTIONS_PATH = "../results/part_1/interactions_clean.csv"
# VIDEOS_PATH = "../data/video.csv"


# # ===============================
# # Imports from your files
# # ===============================

# # --- collaborative.py (ItemCF + SVD) ---
# from collaborative import (
#     load_interactions,
#     build_user_means,
#     build_user_item_centered,
#     build_discounted_item_similarities,
#     recommend_top_n_itemcf,
#     build_rating_matrix,
#     fill_missing_with_item_mean,
#     full_svd,
#     recommend_top_n_svd,
# )

# # --- hybrid.py (content artifacts + switching hybrid) ---
# from hybrid import build_content_based_artifacts, switching_hybrid

# # content_based.py function used by hybrid pipeline
# from content_based import generate_recommendations


# # ===============================
# # 1) Item-based CF
# # ===============================

# def run_itemcf(user_id: int, interactions_df: pd.DataFrame) -> pd.DataFrame:
#     user_mean = build_user_means(interactions_df)
#     user_ratings, user_centered = build_user_item_centered(interactions_df, user_mean)
#     all_items = sorted(interactions_df["video_id"].astype(str).unique().tolist())

#     item_sims = build_discounted_item_similarities(
#         user_centered=user_centered,
#         beta=BETA,
#         min_common=MIN_COMMON,
#     )

#     recs = recommend_top_n_itemcf(
#         user_id=user_id,
#         user_mean=user_mean,
#         user_centered=user_centered,
#         user_ratings=user_ratings,
#         item_sims=item_sims,
#         all_items=all_items,
#         n=TOP_N,
#         k=K_NEIGHBORS,
#         rating_min=RATING_MIN,
#         rating_max=RATING_MAX,
#     )
#     # recs: [(video_id, predicted_rating, neighbors_used), ...] :contentReference[oaicite:3]{index=3}

#     df = pd.DataFrame(recs, columns=["video_id", "predicted_rating", "neighbors_used"])
#     df["method"] = "ItemCF (AdjCos + Discount)"
#     df["history_count"] = int((interactions_df["user_id"] == user_id).sum())
#     return df[["video_id", "predicted_rating", "neighbors_used", "method", "history_count"]]


# # ===============================
# # 2) Truncated SVD
# # ===============================

# def run_truncated_svd(user_id: int, interactions_df: pd.DataFrame) -> pd.DataFrame:
#     R = build_rating_matrix(interactions_df)
#     if user_id not in R.index:
#         return pd.DataFrame(columns=["video_id", "predicted_rating", "method", "history_count"])

#     R_filled = fill_missing_with_item_mean(R)
#     U, Sigma, V = full_svd(R_filled)

#     recs = recommend_top_n_svd(
#         target_user_id=user_id,
#         R=R,
#         R_filled=R_filled,
#         U=U,
#         Sigma=Sigma,
#         V=V,
#         k_latent=K_LATENT,    # truncation happens inside recommend_top_n_svd :contentReference[oaicite:4]{index=4}
#         topn=TOP_N,
#         rating_min=RATING_MIN,
#         rating_max=RATING_MAX,
#     )
#     # recs: [(video_id, predicted_rating), ...]
#     df = pd.DataFrame(recs, columns=["video_id", "predicted_rating"])
#     df["method"] = f"TruncatedSVD (k={K_LATENT})"
#     df["history_count"] = int((interactions_df["user_id"] == user_id).sum())
#     return df[["video_id", "predicted_rating", "method", "history_count"]]


# # ===============================
# # 3) Content-based
# # ===============================

# def run_content_based(user_id: int, interactions_df: pd.DataFrame, videos_df: pd.DataFrame) -> pd.DataFrame:
#     item_matrix, user_profiles, cold_start_vec = build_content_based_artifacts(videos_df, interactions_df)

#     out = generate_recommendations(
#         user_id=user_id,
#         item_matrix=item_matrix,
#         user_profiles=user_profiles,
#         cold_start_vec=cold_start_vec,
#         interactions_df=interactions_df,
#         top_n=TOP_N,
#     )

#     out = out.copy()
#     out["method"] = "ContentBased"
#     out["history_count"] = int((interactions_df["user_id"] == user_id).sum())
#     return out[["video_id", "similarity_score", "user_type", "method", "history_count"]]


# # ===============================
# # 4) Hybrid (Switching)
# # ===============================

# def run_hybrid(user_id: int, interactions_df: pd.DataFrame, videos_df: pd.DataFrame) -> pd.DataFrame:
#     history_count = int((interactions_df["user_id"] == user_id).sum())

#     out = switching_hybrid(
#         user_id=user_id,
#         interactions_df=interactions_df,
#         videos_df=videos_df,
#         threshold=HYBRID_THRESHOLD,
#         top_n=TOP_N,
#         k_latent=K_LATENT,
#     )

#     out = out.copy()

#     # Hybrid SVD branch returns column "score" and method="TruncatedSVD" :contentReference[oaicite:6]{index=6}
#     if "score" in out.columns:
#         out = out.rename(columns={"score": "predicted_rating"})
#         out["method"] = "Hybrid (TruncatedSVD)"
#         out["history_count"] = history_count
#         return out[["video_id", "predicted_rating", "method", "history_count"]]

#     # Content-based branch returns similarity_score and user_type, but may not include method
#     out["method"] = "Hybrid (ContentBased)"
#     out["history_count"] = history_count
#     if "user_type" not in out.columns:
#         out["user_type"] = "Existing" if history_count > 0 else "Cold Start"
#     return out[["video_id", "similarity_score", "user_type", "method", "history_count"]]


# # ===============================
# # Main
# # ===============================

# def main():
#     user_id = int(input("Enter target user_id: ").strip())

#     # Resolve paths relative to main.py location
#     base = Path(__file__).resolve().parent
#     interactions_path = (base / INTERACTIONS_PATH).resolve()
#     videos_path = (base / VIDEOS_PATH).resolve()

#     interactions_df = load_interactions(str(interactions_path))
#     videos_df = pd.read_csv(str(videos_path))

#     print("\n==============================")
#     print(f"TARGET USER: {user_id}")
#     print("==============================")

#     # 1) ItemCF
#     print("\n--- Item-based CF Recommendations ---")
#     itemcf_df = run_itemcf(user_id, interactions_df)
#     print(itemcf_df.to_string(index=False) if not itemcf_df.empty else "(no recommendations)")

#     # 2) Truncated SVD
#     print("\n--- Truncated SVD Recommendations ---")
#     svd_df = run_truncated_svd(user_id, interactions_df)
#     print(svd_df.to_string(index=False) if not svd_df.empty else "(no recommendations)")

#     # 3) Content-based
#     print("\n--- Content-Based Recommendations ---")
#     cb_df = run_content_based(user_id, interactions_df, videos_df)
#     print(cb_df.to_string(index=False) if not cb_df.empty else "(no recommendations)")

#     # 4) Hybrid
#     print("\n--- Hybrid Recommendations ---")
#     hyb_df = run_hybrid(user_id, interactions_df, videos_df)
#     print(hyb_df.to_string(index=False) if not hyb_df.empty else "(no recommendations)")


# if __name__ == "__main__":
#     main()


# main.py  (Streamlit App)
# Run with: streamlit run main.py

import streamlit as st
import pandas as pd
from pathlib import Path

# ===============================
# VARIABLES (edit if needed)
# ===============================
TOP_N = 10
K_LATENT = 20
HYBRID_THRESHOLD = 10

# ItemCF params (match your collaborative.py flow)
K_NEIGHBORS = 250
BETA = 50
MIN_COMMON = 2
RATING_MIN = 1.0
RATING_MAX = 5.0

INTERACTIONS_PATH = "../results/part_1/interactions_clean.csv"
VIDEOS_PATH = "../data/video.csv"


# ===============================
# IMPORTS FROM YOUR FILES
# ===============================
from collaborative import (
    load_interactions,
    build_user_means,
    build_user_item_centered,
    build_discounted_item_similarities,
    recommend_top_n_itemcf,
    build_rating_matrix,
    fill_missing_with_item_mean,
    full_svd,
    recommend_top_n_svd,
)

from hybrid import build_content_based_artifacts, switching_hybrid

from content_based import generate_recommendations


# ===============================
# CACHING LOADS (FAST UI)
# ===============================
@st.cache_data
def load_data(interactions_path: str, videos_path: str):
    interactions_df = load_interactions(interactions_path)
    videos_df = pd.read_csv(videos_path)
    return interactions_df, videos_df


@st.cache_data
def build_content_artifacts(videos_df: pd.DataFrame, interactions_df: pd.DataFrame):
    return build_content_based_artifacts(videos_df, interactions_df)

@st.cache_data
def precompute_itemcf(interactions_df: pd.DataFrame, beta: int, min_common: int):
    user_mean = build_user_means(interactions_df)
    user_ratings, user_centered = build_user_item_centered(interactions_df, user_mean)
    all_items = sorted(interactions_df["video_id"].astype(str).unique().tolist())

    item_sims = build_discounted_item_similarities(
        user_centered=user_centered,
        beta=beta,
        min_common=min_common,
    )
    return user_mean, user_ratings, user_centered, all_items, item_sims

@st.cache_data
def precompute_svd(interactions_df: pd.DataFrame):
    R = build_rating_matrix(interactions_df)
    R_filled = fill_missing_with_item_mean(R)
    U, Sigma, V = full_svd(R_filled)
    return R, R_filled, U, Sigma, V



def show_df(df: pd.DataFrame):
    if df is None or df.empty:
        st.write("(no recommendations)")
        return
    st.markdown(df.to_html(index=False), unsafe_allow_html=True)

# ===============================
# RUNNERS
# ===============================
def run_itemcf(user_id: int, interactions_df: pd.DataFrame) -> pd.DataFrame:
    user_mean, user_ratings, user_centered, all_items, item_sims = precompute_itemcf(
        interactions_df, BETA, MIN_COMMON
    )

    recs = recommend_top_n_itemcf(
        user_id=user_id,
        user_mean=user_mean,
        user_centered=user_centered,
        user_ratings=user_ratings,
        item_sims=item_sims,
        all_items=all_items,
        n=TOP_N,
        k=K_NEIGHBORS,
        rating_min=RATING_MIN,
        rating_max=RATING_MAX,
    )

    df = pd.DataFrame(recs, columns=["video_id", "predicted_rating", "neighbors_used"])
    df["method"] = "ItemCF"
    df["history_count"] = int((interactions_df["user_id"] == user_id).sum())
    return df



def run_truncated_svd(user_id: int, interactions_df: pd.DataFrame) -> pd.DataFrame:
    R, R_filled, U, Sigma, V = precompute_svd(interactions_df)

    if user_id not in R.index:
        return pd.DataFrame(columns=["video_id", "predicted_rating", "method", "history_count"])

    recs = recommend_top_n_svd(
        target_user_id=user_id,
        R=R,
        R_filled=R_filled,
        U=U,
        Sigma=Sigma,
        V=V,
        k_latent=K_LATENT,
        topn=TOP_N,
        rating_min=RATING_MIN,
        rating_max=RATING_MAX,
    )

    df = pd.DataFrame(recs, columns=["video_id", "predicted_rating"])
    df["method"] = f"TruncatedSVD(k={K_LATENT})"
    df["history_count"] = int((interactions_df["user_id"] == user_id).sum())
    return df



def run_content_based(user_id: int, interactions_df: pd.DataFrame, content_artifacts) -> pd.DataFrame:
    item_matrix, user_profiles, cold_start_vec = content_artifacts

    out = generate_recommendations(
        user_id=user_id,
        item_matrix=item_matrix,
        user_profiles=user_profiles,
        cold_start_vec=cold_start_vec,
        interactions_df=interactions_df,
        top_n=TOP_N,
    )

    out = out.copy()
    out["method"] = "ContentBased"
    out["history_count"] = int((interactions_df["user_id"] == user_id).sum())
    return out


def run_hybrid(user_id: int, interactions_df: pd.DataFrame, videos_df: pd.DataFrame) -> pd.DataFrame:
    history_count = int((interactions_df["user_id"] == user_id).sum())

    out = switching_hybrid(
        user_id=user_id,
        interactions_df=interactions_df,
        videos_df=videos_df,
        threshold=HYBRID_THRESHOLD,
        top_n=TOP_N,
        k_latent=K_LATENT,
    ).copy()

    # If SVD branch, it usually returns "score" (predicted rating)
    if "score" in out.columns:
        out = out.rename(columns={"score": "predicted_rating"})
        out["method"] = "Hybrid (TruncatedSVD)"
        out["history_count"] = history_count
        return out[["video_id", "predicted_rating", "method", "history_count"]]

    # Otherwise content-based branch
    out["method"] = "Hybrid (ContentBased)"
    out["history_count"] = history_count
    if "user_type" not in out.columns:
        out["user_type"] = "Existing" if history_count > 0 else "Cold Start"
    return out


# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="Video Recommender Pipeline", layout="wide")

st.title("Short-Form Video Recommendation Pipeline")
st.caption("Runs: Item-based CF • Truncated SVD • Content-based • Hybrid (Switching)")

with st.sidebar:
    st.header("Inputs")
    user_id = st.number_input("Target user_id", min_value=1, value=1, step=1)

    st.header("Settings")
    TOP_N = st.slider("Top-N", 5, 30, TOP_N, 1)
    K_LATENT = st.slider("SVD latent factors (k)", 5, 100, K_LATENT, 1)
    HYBRID_THRESHOLD = st.slider("Hybrid threshold", 1, 50, HYBRID_THRESHOLD, 1)

    run_btn = st.button("Run Recommendations", type="primary")
    
    if st.button("Clear cache (recompute everything)"):
        st.cache_data.clear()
        st.success("Cache cleared. Run again.")


# Resolve data paths relative to this file
base = Path(__file__).resolve().parent
interactions_path = str((base / INTERACTIONS_PATH).resolve())
videos_path = str((base / VIDEOS_PATH).resolve())

# Load once (cached)
try:
    interactions_df, videos_df = load_data(interactions_path, videos_path)
except Exception as e:
    st.error(f"Failed to load data. Check paths.\n\n{e}")
    st.stop()

history_count = int((interactions_df["user_id"] == int(user_id)).sum())
st.info(f"User {int(user_id)} history count = {history_count} | Hybrid uses "
        f"{'Collaborative (SVD)' if history_count >= HYBRID_THRESHOLD else 'Content-Based'} (threshold={HYBRID_THRESHOLD})")

if run_btn:
    # Build content artifacts once (cached)
    try:
        content_artifacts = build_content_artifacts(videos_df, interactions_df)
    except Exception as e:
        st.error(f"Failed to build content-based artifacts.\n\n{e}")
        st.stop()

    # col1, col2 = st.columns(2)
    # col3, col4 = st.columns(2)

    with st.spinner("Running ItemCF..."):
        try:
            itemcf_df = run_itemcf(int(user_id), interactions_df)
        except Exception as e:
            itemcf_df = pd.DataFrame()
            st.error(f"ItemCF failed: {e}")

    with st.spinner("Running Truncated SVD..."):
        try:
            svd_df = run_truncated_svd(int(user_id), interactions_df)
        except Exception as e:
            svd_df = pd.DataFrame()
            st.error(f"SVD failed: {e}")

    with st.spinner("Running Content-Based..."):
        try:
            cb_df = run_content_based(int(user_id), interactions_df, content_artifacts)
        except Exception as e:
            cb_df = pd.DataFrame()
            st.error(f"Content-based failed: {e}")

    with st.spinner("Running Hybrid..."):
        try:
            hyb_df = run_hybrid(int(user_id), interactions_df, videos_df)
        except Exception as e:
            hyb_df = pd.DataFrame()
            st.error(f"Hybrid failed: {e}")

    st.subheader("Item-based CF")
    show_df(itemcf_df)

    st.divider()

    st.subheader("Truncated SVD")
    show_df(svd_df)

    st.divider()

    st.subheader("Content-based")
    show_df(cb_df)

    st.divider()

    st.subheader("Hybrid (Switching)")
    show_df(hyb_df)

else:
    st.warning("Set a target user_id and click **Run Recommendations**.")
