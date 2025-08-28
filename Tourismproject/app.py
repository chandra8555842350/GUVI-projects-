import streamlit as st 
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------------------
# Load
# ------------------------------
df = pd.read_csv("cleanedtourism.csv")
print(df.head())

# ------------------------------
# Prepare matrices
# ------------------------------
# User–Item ratings matrix
user_item_matrix = df.pivot_table( index='UserId', columns='AttractionId', values='Rating').fillna(0)

# Sparse, float32 to save memory
user_item_sparse = csr_matrix(user_item_matrix.values.astype(np.float32))

# Basic popularity fallback (global mean rating per item)
item_popularity = (
    df.groupby('AttractionId')['Rating'].mean()
      .reindex(user_item_matrix.columns)
      .fillna(0)
)

# ------------------------------
# Collaborative Filtering (KNN, no full similarity matrix)
# ------------------------------
n_users = user_item_matrix.shape[0]

# Fit a user KNN model (cosine distance)
user_knn = NearestNeighbors(
    metric='cosine', algorithm='brute'
).fit(user_item_sparse)

def recommend_collaborative(user_id, top_n=5, k_neighbors=50):
    if user_id not in user_item_matrix.index:
        # cold-start user → popular items
        return item_popularity.sort_values(ascending=False).head(top_n).index.tolist()

    user_idx = user_item_matrix.index.get_loc(user_id)
    # Get k neighbors (cosine distance in [0, 1]); similarity = 1 - distance
    k = min(k_neighbors + 1, n_users)  # +1 for the user itself
    distances, indices = user_knn.kneighbors(user_item_sparse[user_idx], n_neighbors=k)
    distances = distances.ravel()
    indices = indices.ravel()

    # Drop self index if present
    mask = indices != user_idx
    neighbor_indices = indices[mask]
    sims = 1.0 - distances[mask]  # convert cosine distance → similarity

    if neighbor_indices.size == 0 or sims.sum() == 0:
        return item_popularity.sort_values(ascending=False).head(top_n).index.tolist()

    # Weighted average of neighbor ratings
    neighbor_ratings = user_item_matrix.iloc[neighbor_indices].values  # shape: (k, n_items)
    weights = sims[:, None]  # (k, 1)
    weighted_scores = (neighbor_ratings * weights).sum(axis=0) / (weights.sum(axis=0) + 1e-8)

    # Remove already visited items
    user_vector = user_item_matrix.iloc[user_idx].values  # (n_items,)
    candidate_scores = np.where(user_vector > 0, -np.inf, weighted_scores)

    # Top-N indices
    if np.all(~np.isfinite(candidate_scores)):
        return item_popularity.sort_values(ascending=False).head(top_n).index.tolist()

    top_idx = np.argpartition(-candidate_scores, range(min(top_n, candidate_scores.size)))[:top_n]
    top_idx = top_idx[np.argsort(-candidate_scores[top_idx])]
    return user_item_matrix.columns[top_idx].tolist()

# ------------------------------
# Content-Based (Item KNN on TF-IDF)
# ------------------------------
# Create simple text features for attractions
df['combined_features'] = (
    df['AttractionTypeId'].astype(str) + " " + df['AttractionCityId'].astype(str)
)

attraction_features = (
    df[['AttractionId', 'combined_features']]
    .drop_duplicates()
    .set_index('AttractionId')
)

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(attraction_features['combined_features'])  # sparse

# Fit item KNN on TF-IDF
item_knn = NearestNeighbors(metric='cosine', algorithm='brute').fit(tfidf_matrix)

# Helper to get similar items (returns dict: item_id -> similarity)
def similar_items(attraction_id, k_neighbors=50):
    if attraction_id not in attraction_features.index:
        return {}
    idx = attraction_features.index.get_loc(attraction_id)
    k = min(k_neighbors + 1, tfidf_matrix.shape[0])  # +1 to include the item itself
    distances, indices = item_knn.kneighbors(tfidf_matrix[idx], n_neighbors=k)
    distances = distances.ravel()
    indices = indices.ravel()

    # Exclude the item itself
    mask = indices != idx
    indices = indices[mask]
    sims = 1.0 - distances[mask]

    ids = attraction_features.index[indices]
    return dict(zip(ids, sims))

def recommend_content(user_id, top_n=5, k_neighbors=50):
    if user_id not in user_item_matrix.index:
        return item_popularity.sort_values(ascending=False).head(top_n).index.tolist()

    # Items the user has interacted with
    user_row = user_item_matrix.loc[user_id]
    visited = user_row[user_row > 0].index.tolist()
    if not visited:
        return item_popularity.sort_values(ascending=False).head(top_n).index.tolist()

    # Accumulate similarity scores from neighbors of visited items
    scores = {}
    for aid in visited:
        for sid, sim in similar_items(aid, k_neighbors=k_neighbors).items():
            if sid in visited:
                continue
            scores[sid] = scores.get(sid, 0.0) + sim

    if not scores:
        return item_popularity.sort_values(ascending=False).head(top_n).index.tolist()

    # Rank by accumulated similarity
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [aid for aid, _ in ranked[:top_n]]

# ------------------------------
# Hybrid
# ------------------------------
def recommend_hybrid(user_id, top_n=5):
    collab = recommend_collaborative(user_id, top_n=top_n*2, k_neighbors=50)
    content = recommend_content(user_id, top_n=top_n*2, k_neighbors=50)

    combined = []
    seen = set()
    for lst in (collab, content, item_popularity.index.tolist()):
        for aid in lst:
            if aid not in seen:
                combined.append(aid)
                seen.add(aid)
            if len(combined) >= top_n:
                return combined
    return combined[:top_n]

# ------------------------------
# Streamlit App
# ------------------------------
st.title("🎢 Attraction Recommendation System")

# Sidebar Controls
st.sidebar.header("Options")
user_id = st.sidebar.selectbox("Select User ID", sorted(df['UserId'].unique()))
top_n = st.sidebar.slider("Number of Recommendations", 1, 10, 5)
method = st.sidebar.radio("Select Recommendation Method", ["Collaborative", "Content-Based", "Hybrid"])

# Run Recommendation
if method == "Collaborative":
    recs = recommend_collaborative(user_id, top_n)
elif method == "Content-Based":
    recs = recommend_content(user_id, top_n)
else:
    recs = recommend_hybrid(user_id, top_n)

# Show Results
st.subheader(f"Recommended Attractions for User {user_id} ({method}):")
if recs:
    st.success(recs)
else:
    st.warning("No recommendations found for this user.")
