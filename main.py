
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import html

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ==============================
# LOAD MODEL
# ==============================

@st.cache_resource
def load_data():

    with open("df.pkl", "rb") as f:
        df = pickle.load(f)

    with open("indices.pkl", "rb") as f:
        indices = pickle.load(f)

    with open("tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)

    return df, indices, tfidf_matrix


try:
    df, indices, tfidf_matrix = load_data()
except Exception as e:
    st.error(f"Unable to load model files: {e}")
    st.stop()

if not isinstance(df, pd.DataFrame):
    df = pd.DataFrame(df)

df = df.reset_index(drop=True)

if "title" not in df.columns:
    st.error("Your dataframe must contain a 'title' column.")
    st.stop()

if len(df) != tfidf_matrix.shape[0]:
    st.error("Movie dataset and TF-IDF matrix have different row counts.")
    st.stop()

# ==============================
# CUSTOM CSS
# ==============================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: #0b0b0f;
    color: white;
}

header {
    visibility: hidden;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
}

.hero {
    text-align: center;
    padding: 35px 0 25px;
}

.logo {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: 2px;
    color: white;
}

.logo span {
    color: #e50914;
}

.tagline {
    color: #a5a5a5;
    font-size: 15px;
}

.section {
    font-size: 25px;
    font-weight: 700;
    margin: 30px 0 18px;
    color: white;
}

.movie-card {
    background: #17171f;
    padding: 12px;
    border-radius: 12px;
    border: 1px solid #292933;
    min-height: 75px;
}

.movie-name {
    color: white;
    font-size: 14px;
    font-weight: 600;
    min-height: 42px;
}

.movie-score {
    color: #e50914;
    font-size: 12px;
}

.stButton > button {
    background: #e50914;
    color: white;
    border: none;
    border-radius: 8px;
    height: 45px;
    font-weight: 600;
}

.stButton > button:hover {
    background: #b20710;
    color: white;
}

.stSelectbox div[data-baseweb="select"] {
    background: #17171f;
}

div[data-testid="stImage"] img {
    border-radius: 10px;
}

.footer {
    text-align: center;
    color: #777;
    padding: 35px 0 10px;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)

# ==============================
# RECOMMENDATION FUNCTION
# ==============================

def recommend(movie):

    movie_index = indices.get(movie)

    if movie_index is None:
        return []

    if isinstance(movie_index, (list, tuple, np.ndarray)):
        movie_index = movie_index[0]

    if isinstance(movie_index, pd.Series):
        movie_index = movie_index.iloc[0]

    movie_index = int(movie_index)

    if not 0 <= movie_index < len(df):
        return []

    scores = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix
    ).flatten()

    top_indices = np.argsort(scores)[::-1]

    recommendations = []

    for i in top_indices:

        if i == movie_index:
            continue

        recommendations.append({
            "movie": df.iloc[i],
            "score": scores[i]
        })

        if len(recommendations) == 5:
            break

    return recommendations


# ==============================
# POSTER FUNCTION
# ==============================

def get_poster(movie):

    for column in ["poster_url", "poster_path", "poster"]:

        if column not in movie.index:
            continue

        poster = movie[column]

        if pd.isna(poster):
            continue

        poster = str(poster).strip()

        if poster.startswith("http"):
            return poster

        if poster.startswith("/"):
            return "https://image.tmdb.org/t/p/w500" + poster

    return None


# ==============================
# HEADER
# ==============================

st.markdown("""
<div class="hero">
    <div class="logo">CINE<span>MATCH</span> 🎬</div>
    <div class="tagline">
        Discover your next favourite movie.
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================
# SEARCH SECTION
# ==============================

st.markdown(
    '<div class="section">Find Your Movie</div>',
    unsafe_allow_html=True
)

movies = sorted(
    df["title"].dropna().astype(str).unique().tolist()
)

col1, col2 = st.columns([4, 1])

with col1:

    selected_movie = st.selectbox(
        "Search movies",
        movies,
        index=None,
        placeholder="Type a movie name..."
    )

with col2:

    st.write("")
    st.write("")

    search = st.button(
        "Get Recommendations",
        use_container_width=True
    )

# ==============================
# SHOW RECOMMENDATIONS
# ==============================

if search:

    if not selected_movie:

        st.warning("Please select a movie first.")

    else:

        with st.spinner("Finding similar movies..."):

            results = recommend(selected_movie)

        if results:

            st.markdown(
                f'<div class="section">Movies Similar to {html.escape(selected_movie)}</div>',
                unsafe_allow_html=True
            )

            cols = st.columns(5)

            for col, result in zip(cols, results):

                movie = result["movie"]
                score = result["score"]

                title = str(movie["title"])
                poster = get_poster(movie)

                with col:

                    if poster:
                        st.image(
                            poster,
                            use_container_width=True
                        )
                    else:
                        st.markdown("""
                        <div style="
                            height:250px;
                            background:#20202a;
                            border-radius:10px;
                            display:flex;
                            align-items:center;
                            justify-content:center;
                            font-size:45px;">
                            🎬
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown(
                        f"""
                        <div class="movie-card">
                            <div class="movie-name">
                                {html.escape(title)}
                            </div>
                            <div class="movie-score">
                                Similarity: {score:.1%}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        else:
            st.error("No recommendations found.")

# ==============================
# FOOTER
# ==============================

st.markdown("""
<div class="footer">
    CINE MATCH © 2026 | Powered by Machine Learning
</div>
""", unsafe_allow_html=True)
