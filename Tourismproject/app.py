import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------------------
# Example Data (Replace with your real dataset)
# ------------------------------
data = {
    "UserId": [1, 1, 2, 2, 3, 3, 4, 5],
    "AttractionId": [101, 102, 101, 103, 102, 104, 103, 104],
    "Rating": [5, 3, 4, 2, 5, 4, 3, 4],
    "AttractionTypeId": [1, 2, 1, 3, 2, 3, 3, 3],
    "AttractionCityId": [10, 20, 10, 30, 20, 30, 30, 30],
}
df = pd.DataFrame(data)

# ------------------------------
# Collaborative Filtering
# ------------------------------
user_item_matrix = df.pivot_table(index='UserId', columns='AttractionId', values='Rating').fillna(0)

user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(
    user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index
)

def recommend_collaborative(user_id, top_n=5):
    if user_id not in user_similarity_df.index:
        return []
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]
    similar_users_ratings = user_item_matrix.loc[similar_users].mean(axis=0)
    visited = user_item_matrix.loc[user_id]
    similar_users_ratings = similar_users_ratings[visited == 0]
    return similar_users_ratings.sort_values(ascending=False).head(top_n).index.tolist()

# ------------------------------
# Content-Based Filtering
# ------------------------------
df['combined_features'] = df['AttractionTypeId'].astype(str) + " " + df['AttractionCityId'].astype(str)

attraction_features = df[['AttractionId', 'combined_features']].drop_duplicates().set_index('AttractionId')

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(attraction_features['combined_features'])

attraction_similarity = cosine_similarity(tfidf_matrix)
attraction_similarity_df = pd.DataFrame(
    attraction_similarity, index=attraction_features.index, columns=attraction_features.index
)

def recommend_content(user_id, top_n=5):
    if user_id not in user_item_matrix.index:
        return []
    visited_attractions = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index.tolist()
    scores = pd.Series(dtype=float)
    for attraction in visited_attractions:
        scores = scores.add(pd.Series(attraction_similarity_df[attraction]), fill_value=0)
    scores = scores.drop(visited_attractions, errors='ignore')
    return scores.sort_values(ascending=False).head(top_n).index.tolist()

# ------------------------------
# Hybrid Recommendation
# ------------------------------
def recommend_hybrid(user_id, top_n=5):
    collab_recs = recommend_collaborative(user_id, top_n * 2)
    content_recs = recommend_content(user_id, top_n * 2)
    combined = list(dict.fromkeys(collab_recs + content_recs))  # keep order, remove duplicates
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
