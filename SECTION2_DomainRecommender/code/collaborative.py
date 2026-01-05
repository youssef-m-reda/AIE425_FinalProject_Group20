from __future__ import annotations

import os
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd


# ============================================================
# Shared: relative dataset path (from /code/)
# ============================================================
DATA_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "results","part_1", "interactions_clean.csv")
)


# ============================================================
# (A) Item-based CF: Adjusted Cosine + Discounted Similarity
# (from your collaborative.py logic)
# ============================================================
def load_interactions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {"user_id", "video_id", "rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    df = df.dropna(subset=["user_id", "video_id", "rating"]).copy()
    df["user_id"] = df["user_id"].astype(int)
    df["video_id"] = df["video_id"].astype(str)
    df["rating"] = df["rating"].astype(float)
    return df


def build_user_means(df: pd.DataFrame) -> Dict[int, float]:
    return df.groupby("user_id")["rating"].mean().to_dict()


def build_user_item_centered(
    df: pd.DataFrame,
    user_mean: Dict[int, float],
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]:
    user_ratings: Dict[int, Dict[str, float]] = defaultdict(dict)
    user_centered: Dict[int, Dict[str, float]] = defaultdict(dict)

    for row in df.itertuples(index=False):
        u = int(row.user_id)
        i = str(row.video_id)
        r = float(row.rating)
        user_ratings[u][i] = r
        user_centered[u][i] = r - user_mean[u]

    return user_ratings, user_centered


def build_discounted_item_similarities(
    user_centered: Dict[int, Dict[str, float]],
    beta: int = 50,
    min_common: int = 2,
) -> Dict[str, List[Tuple[str, float, int]]]:
    numer: Dict[Tuple[str, str], float] = defaultdict(float)
    den_i: Dict[Tuple[str, str], float] = defaultdict(float)
    den_j: Dict[Tuple[str, str], float] = defaultdict(float)
    cnt: Dict[Tuple[str, str], int] = defaultdict(int)

    for _, items_dict in user_centered.items():
        items = list(items_dict.items())
        n = len(items)
        if n < 2:
            continue

        for a in range(n):
            i, s_i = items[a]
            for b in range(a + 1, n):
                j, s_j = items[b]

                if i < j:
                    key = (i, j)
                    numer[key] += s_i * s_j
                    den_i[key] += s_i * s_i
                    den_j[key] += s_j * s_j
                    cnt[key] += 1
                else:
                    key = (j, i)
                    numer[key] += s_j * s_i
                    den_i[key] += s_j * s_j
                    den_j[key] += s_i * s_i
                    cnt[key] += 1

    sims: Dict[str, List[Tuple[str, float, int]]] = defaultdict(list)

    for (i, j), num in numer.items():
        common = cnt[(i, j)]
        if common < min_common:
            continue

        di = den_i[(i, j)]
        dj = den_j[(i, j)]
        if di <= 0.0 or dj <= 0.0:
            continue

        adj_cos = num / (math.sqrt(di) * math.sqrt(dj))
        dfactor = (min(common, beta) / float(beta)) if beta > 0 else 1.0
        ds = adj_cos * dfactor

        sims[i].append((j, ds, common))
        sims[j].append((i, ds, common))

    return sims


def predict_rating_itemcf(
    user_id: int,
    target_item: str,
    user_mean: Dict[int, float],
    user_centered: Dict[int, Dict[str, float]],
    user_ratings: Dict[int, Dict[str, float]],
    item_sims: Dict[str, List[Tuple[str, float, int]]],
    k: int = 20,
    rating_min: float = 1.0,
    rating_max: float = 5.0,
) -> Optional[Tuple[float, int]]:
    """
    pred(u,t) = r̄(u) + Σ DS(j,t)*s(u,j) / Σ |DS(j,t)|
    where j are top-k most similar items to t that user u has rated.

    Returns:
      (predicted_rating, neighbors_used)

    neighbors_used = number of neighbor items actually used in the numerator/denominator.
    If neighbors_used == 0 => fallback to user mean (no similarity evidence).
    """
    if user_id not in user_centered:
        return None

    # If already rated, return actual rating (neighbors_used=0 because no prediction step)
    if target_item in user_ratings[user_id]:
        r = float(user_ratings[user_id][target_item])
        r = max(rating_min, min(rating_max, r))
        return r, 0

    mu = user_mean.get(user_id, 0.0)
    rated_items = user_centered[user_id]  # {item: s(u,item)}

    # Collect neighbor similarities for items the user already rated
    neighbors: List[Tuple[str, float]] = []
    for (j, ds, _) in item_sims.get(target_item, []):
        if j in rated_items:
            neighbors.append((j, ds))

    if not neighbors:
        # No similarity evidence -> fallback to user mean
        pred = float(mu)
        pred = max(rating_min, min(rating_max, pred))
        return pred, 0

    # Top-k by similarity
    neighbors.sort(key=lambda x: x[1], reverse=True)
    neighbors = neighbors[:k]

    num = 0.0
    den = 0.0
    for j, ds in neighbors:
        num += ds * rated_items[j]   # ds(j,t) * s(u,j)
        den += abs(ds)

    if den == 0.0:
        pred = float(mu)
        pred = max(rating_min, min(rating_max, pred))
        return pred, 0

    pred = mu + (num / den)
    pred = max(rating_min, min(rating_max, pred))
    return float(pred), len(neighbors)



def recommend_top_n_itemcf(
    user_id: int,
    user_mean: Dict[int, float],
    user_centered: Dict[int, Dict[str, float]],
    user_ratings: Dict[int, Dict[str, float]],
    item_sims: Dict[str, List[Tuple[str, float, int]]],
    all_items: List[str],
    n: int = 10,
    k: int = 20,
    rating_min: float = 1.0,
    rating_max: float = 5.0,
) -> List[Tuple[str, float, int]]:
    """
    Recommend Top-N unseen items for a user by predicted rating.

    Returns:
      [(video_id, predicted_rating, neighbors_used), ...]

    Set `SKIP_FALLBACK = True` to exclude items predicted with 0 neighbors_used.
    """
    if user_id not in user_ratings:
        return []

    # Toggle this based on what you want to observe:
    # True  => only "evidence-based" (neighbors_used > 0)
    # False => allow fallback-to-mean items too
    SKIP_FALLBACK = True

    seen = set(user_ratings[user_id].keys())
    candidates = [it for it in all_items if it not in seen]

    preds: List[Tuple[str, float, int]] = []

    for it in candidates:
        out = predict_rating_itemcf(
            user_id=user_id,
            target_item=it,
            user_mean=user_mean,
            user_centered=user_centered,
            user_ratings=user_ratings,
            item_sims=item_sims,
            k=k,
            rating_min=rating_min,
            rating_max=rating_max,
        )
        if out is None:
            continue

        score, used_n = out
        if used_n < 2: 
            continue


        preds.append((it, score, used_n))

    preds.sort(key=lambda x: x[1], reverse=True)
    return preds[:n]


# ============================================================
# (B) SVD Recommender (NO metrics)
# (adapted from your svd.py: only factorization + recommend)
# ============================================================
def build_rating_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(index="user_id", columns="video_id", values="rating", aggfunc="mean")


def fill_missing_with_item_mean(R: pd.DataFrame) -> pd.DataFrame:
    item_means = R.mean(axis=0)
    return R.fillna(item_means)


def full_svd(R_filled: pd.DataFrame):
    Rm = R_filled.values
    m, n = Rm.shape

    RtR = Rm.T @ Rm
    eigenvalues, V = np.linalg.eigh(RtR)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]

    singular_values = np.sqrt(np.maximum(eigenvalues, 0.0))

    Sigma = np.zeros((m, n), dtype=float)
    np.fill_diagonal(Sigma, singular_values)

    U = np.zeros((m, m), dtype=float)
    for i in range(min(m, n)):
        if singular_values[i] > 1e-10:
            U[:, i] = (Rm @ V[:, i]) / singular_values[i]

    return U, Sigma, V


def recommend_top_n_svd(
    target_user_id: int,
    R: pd.DataFrame,
    R_filled: pd.DataFrame,
    U: np.ndarray,
    Sigma: np.ndarray,
    V: np.ndarray,
    k_latent: int = 20,
    topn: int = 10,
    rating_min: float = 1.0,
    rating_max: float = 5.0,
) -> List[Tuple[str, float]]:
    if target_user_id not in R.index:
        return []

    Uk = U[:, :k_latent]
    Sk = Sigma[:k_latent, :k_latent]
    Vk = V[:, :k_latent]

    u_idx = R.index.get_loc(target_user_id)

    # Predict all items for this user: (u_row * Sigma) @ V^T
    user_latent_vector = Uk[u_idx, :] @ Sk
    preds = user_latent_vector @ Vk.T  # shape: (n_items,)

    pred_series = pd.Series(preds, index=R.columns)

    # Remove already-rated items (use original sparse R)
    seen = set(R.loc[target_user_id].dropna().index)
    pred_series = pred_series.drop(labels=list(seen), errors="ignore")

    # Clamp to rating scale (optional)
    pred_series = pred_series.clip(lower=rating_min, upper=rating_max)

    top = pred_series.sort_values(ascending=False).head(topn)
    return list(zip(top.index.astype(str).tolist(), top.values.astype(float).tolist()))


# ============================================================
# Compare + print differences
# ============================================================
def pretty_print(title: str, recs: List[Tuple[str, float]]):
    print(f"\n{title}")
    if not recs:
        print("  (no recommendations)")
        return
    for i, (item, score) in enumerate(recs, start=1):
        print(f"  {i:>2}. video_id={item} | score={score:.4f}")


def compare_lists(a: List[Tuple[str, float]], b: List[Tuple[str, float]]):
    a_items = [x[0] for x in a]
    b_items = [x[0] for x in b]

    set_a: Set[str] = set(a_items)
    set_b: Set[str] = set(b_items)

    overlap = list(set_a & set_b)
    only_a = [x for x in a_items if x not in set_b]
    only_b = [x for x in b_items if x not in set_a]

    print("\n=== Comparison ===")
    print(f"Overlap count: {len(overlap)}")
    if overlap:
        print("Overlap items:", overlap[:20])

    print(f"Only ItemCF count: {len(only_a)}")
    if only_a:
        print("Only ItemCF:", only_a[:20])

    print(f"Only SVD count: {len(only_b)}")
    if only_b:
        print("Only SVD:", only_b[:20])


def main():
    # ======================
    # SETTINGS (edit here)
    # ======================
    target_user_id = 1
    topn = 50

    # ItemCF hyperparams (slide-aligned)
    k_neighbors = 250
    beta = 50
    min_common = 2

    # SVD hyperparams
    k_latent = 20

    rating_min = 1.0
    rating_max = 5.0
    # ======================

    print("Loading interactions from:", DATA_PATH)
    df = load_interactions(DATA_PATH)
    print(f"Interactions: {len(df):,} | Users: {df['user_id'].nunique():,} | Items: {df['video_id'].nunique():,}")

    # ---------- ItemCF ----------
    user_mean = build_user_means(df)
    user_ratings, user_centered = build_user_item_centered(df, user_mean)
    all_items = sorted(df["video_id"].astype(str).unique().tolist())

    print("\n[ItemCF] Building discounted item similarities...")
    item_sims = build_discounted_item_similarities(
        user_centered=user_centered,
        beta=beta,
        min_common=min_common,
    )

    recs_itemcf = recommend_top_n_itemcf(
        user_id=target_user_id,
        user_mean=user_mean,
        user_centered=user_centered,
        user_ratings=user_ratings,
        item_sims=item_sims,
        all_items=all_items,
        n=topn,
        k=k_neighbors,
        rating_min=rating_min,
        rating_max=rating_max,
    )

    # ---------- SVD ----------
    print("\n[SVD] Building rating matrix + filling with item means...")
    R = build_rating_matrix(df)
    R_filled = fill_missing_with_item_mean(R)

    print("[SVD] Running full SVD (may take time depending on matrix size)...")
    U, Sigma, V = full_svd(R_filled)

    recs_svd = recommend_top_n_svd(
        target_user_id=target_user_id,
        R=R,
        R_filled=R_filled,
        U=U,
        Sigma=Sigma,
        V=V,
        k_latent=k_latent,
        topn=topn,
        rating_min=rating_min,
        rating_max=rating_max,
    )

    # ---------- Print ----------
    print(f"\nTARGET USER: {target_user_id}")
    # pretty_print("Top-N ItemCF (Adjusted Cosine + Discount)", recs_itemcf)
    print("\nTop-N ItemCF (Adjusted Cosine + Discount)")
    for rank, (vid, score, used_n) in enumerate(recs_itemcf, start=1):
        print(f"  {rank:>2}. video_id={vid} | score={score:.4f} | neighbors_used={used_n}")

    pretty_print("Top-N SVD", recs_svd)

    compare_lists(recs_itemcf, recs_svd)


if __name__ == "__main__":
    main()
    