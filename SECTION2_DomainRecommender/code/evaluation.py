import pandas as pd
import numpy as np
import random
from hybrid import switching_hybrid, build_content_based_artifacts
from collaborative import load_interactions

def train_test_split_by_user(df, test_ratio=0.2):
    train_list, test_list = [], []
    for user_id, group in df.groupby('user_id'):
        if len(group) < 2:
            train_list.append(group)
            continue
        shuffled = group.sample(frac=1, random_state=42)
        n_test = max(1, int(len(group) * test_ratio))
        test_list.append(shuffled.head(n_test))
        train_list.append(shuffled.tail(len(group) - n_test))
    return pd.concat(train_list), pd.concat(test_list)

import pandas as pd
import numpy as np
import random
# Ensure these imports match your file structure
from hybrid import switching_hybrid, build_content_based_artifacts
from collaborative import load_interactions

def calculate_dual_metrics(rec_ids, test_ids, videos_df):
    """
    Calculates both Exact Match (Strict) and Category Match (Soft).
    Returns: (Exact Precision, Category Precision)
    """
    # 1. FORCE STRING CONVERSION
    rec_ids = [str(x) for x in rec_ids]
    test_ids = [str(x) for x in test_ids]
    
    # 2. Safety Check
    if not rec_ids or not test_ids:
        return 0.0, 0.0
    
    # 3. EXACT MATCH (Did they watch this specific video?)
    exact_hits = len(set(rec_ids).intersection(set(test_ids)))
    exact_precision = exact_hits / len(rec_ids)
    
    # 4. CATEGORY MATCH (Did they watch a video in this genre?)
    # Get categories for recommended items
    # Note: Ensure your videos_df has a 'category_id' or 'category' column. 
    # If it is named 'genre', change 'category_id' below to 'genre'.
    rec_cats = videos_df[videos_df['video_id'].isin(rec_ids)]['genre'].tolist()
    test_cats = videos_df[videos_df['video_id'].isin(test_ids)]['genre'].tolist()
    
    test_cat_set = set(test_cats)
    soft_hits = 0
    for cat in rec_cats:
        if cat in test_cat_set:
            soft_hits += 1
            
    soft_precision = soft_hits / len(rec_ids)

    return exact_precision, soft_precision

def main():
    interactions_df = load_interactions("../results/part_1/interactions_clean.csv")
    videos_df = pd.read_csv("../data/video.csv")
    
    # Standardize Video IDs
    if "row_id" in videos_df.columns: videos_df = videos_df.rename(columns={"row_id": "video_id"})
    videos_df['video_id'] = videos_df['video_id'].astype(str)
    
    # Initialize Models
    item_matrix, _, _ = build_content_based_artifacts(videos_df, interactions_df)
    train_df, test_df = train_test_split_by_user(interactions_df)
    
    # Select users for evaluation
    eval_users = test_df['user_id'].unique()[:30]
    results = []

    print(f"Starting evaluation on {len(eval_users)} users...")

    for user_id in eval_users:
        test_ids = test_df[test_df['user_id'] == user_id]['video_id'].tolist()
        
        # Skip users with empty test sets to avoid errors
        if not test_ids:
            continue
        
        # 1. Hybrid (Switching between Content & SVD)
        h_recs = switching_hybrid(user_id, train_df, videos_df, threshold=5, top_n=10)['video_id'].tolist()
        
        # 2. Pure Content-Based (Threshold high to force Content path)
        c_recs = switching_hybrid(user_id, train_df, videos_df, threshold=9999, top_n=10)['video_id'].tolist()
        
        # 3. Popularity
        p_recs = videos_df.nlargest(10, 'views')['video_id'].tolist()
        
        # 4. Random
        r_recs = random.sample(list(videos_df['video_id'].unique()), 10)
        
        # --- CALCULATE METRICS (Exact & Soft) ---
        h_exact, h_soft = calculate_dual_metrics(h_recs, test_ids, videos_df)
        c_exact, c_soft = calculate_dual_metrics(c_recs, test_ids, videos_df)
        p_exact, p_soft = calculate_dual_metrics(p_recs, test_ids, videos_df)
        r_exact, r_soft = calculate_dual_metrics(r_recs, test_ids, videos_df)
        
        results.append({
            'Hybrid_Exact': h_exact, 'Hybrid_Soft': h_soft,
            'Content_Exact': c_exact, 'Content_Soft': c_soft,
            'Pop_Exact': p_exact, 'Pop_Soft': p_soft,
            'Rand_Exact': r_exact, 'Rand_Soft': r_soft
        })

    # Create Summary Table
    summary = pd.DataFrame(results).mean()
    
    final_df = pd.DataFrame({
        'Approach': ['Hybrid (Switching)', 'Pure Content-Based', 'Most Popular', 'Random'],
        'Exact Precision@10': [summary['Hybrid_Exact'], summary['Content_Exact'], summary['Pop_Exact'], summary['Rand_Exact']],
        'Category Precision@10': [summary['Hybrid_Soft'], summary['Content_Soft'], summary['Pop_Soft'], summary['Rand_Soft']]
    })
    
    print("\n=== FINAL COMPREHENSIVE BASELINE TABLE ===\n")
    print(final_df.to_string(index=False))

if __name__ == "__main__":
    main()