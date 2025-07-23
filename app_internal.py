import streamlit as st
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

st.set_page_config(page_title="SELECT COURSES IN A CLICK !", page_icon="📘", layout="wide")

# ---------- TEXT CLEANING  ---------- #
def simple_clean(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    return ' '.join(tokens)

# ---------- LOAD & PREPARE DATA ---------- #

def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        expected_cols = [
            'id', 'title', 'url', 'is_paid', 'instructor_names', 'category',
            'headline', 'num_subscribers', 'rating', 'num_reviews',
            'instructional_level', 'objectives', 'curriculum'
        ]
        if df.shape[1] == len(expected_cols):
            df.columns = expected_cols
        df['combined'] = df['headline'].fillna('') + ' ' + df['objectives'].fillna('') + ' ' + df['curriculum'].fillna('')
        df['cleaned'] = df['combined'].apply(simple_clean)
        return df
    else:
        return pd.DataFrame()

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def encode_courses(_model, texts):
    def batch_encode(texts, batch_size=32):  # smaller batch
        all_embeddings = []
        total = len(texts)
        for i in range(0, total, batch_size):
            batch = texts[i:i+batch_size]
            emb = _model.encode(batch, show_progress_bar=True)  # enable progress
            all_embeddings.extend(emb)
        return np.array(all_embeddings)
    return batch_encode(texts)

# ---------- RECOMMENDATION FUNCTION ---------- #
def recommend_courses(user_input, df, embeddings, model, top_n=5):
    user_input_clean = simple_clean(user_input)
    user_embedding = model.encode([user_input_clean])
    similarities = cosine_similarity(user_embedding, embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_n]
    recommended = df.iloc[top_indices][['title', 'url']].copy()
    recommended['score'] = similarities[top_indices]
    return recommended.reset_index(drop=True)

# ---------- MAIN UI ---------- #
def main():
    st.markdown("""
        <style>
            .big-title {
                font-size: 3.5rem;
                font-weight: 800;
                text-align: center;
                background: -webkit-linear-gradient(45deg, #00f5ff, #ff00ff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.2em;
            }
            .sub-title {
                text-align: center;
                font-size: 1.2rem;
                color: #a0a0a0;
                margin-bottom: 2em;
            }
            .course-card {
                background-color: #1a1a2e;
                padding: 1.5rem;
                margin-bottom: 1rem;
                border-radius: 1rem;
                box-shadow: 0 10px 20px rgba(0,0,0,0.3);
                transition: 0.3s ease-in-out;
            }
            .course-card:hover {
                transform: scale(1.02);
                box-shadow: 0 20px 40px rgba(0,0,0,0.4);
            }
            .score {
                background-color: #00f5ff33;
                padding: 0.5rem 1rem;
                border-radius: 1rem;
                font-weight: bold;
                display: inline-block;
                margin-top: 0.5rem;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""<div class='big-title'>SELECT COURSES IN A CLICK !</div>""", unsafe_allow_html=True)
    st.markdown("""<div class='sub-title'> Smart Course Recommendations</div>""", unsafe_allow_html=True)

    # File upload and data load
    uploaded_file = st.file_uploader(" Upload your CSV file", type=["csv"])
    df = load_data(uploaded_file)  

    if df.empty:
        return

    model = load_model()
    embeddings = encode_courses(model, df['cleaned'].tolist())

    user_query = st.text_input(" Describe your learning goal:", "python data science for beginners")
    if st.button(" Get Recommendations"):
        with st.spinner("Finding best matches..."):
            time.sleep(1.5)
            results = recommend_courses(user_query, df, embeddings, model)
            st.success("Top Matches")

            for _, row in results.iterrows():
                st.markdown(f"""
                    <div class='course-card'>
                        <h4>{row['title']}</h4>
                        <div class='score'>Similarity Score: {row['score']:.3f}</div><br><br>
                        <a href='{row['url']}' target='_blank'>🔗 View Course</a>
                    </div>
                """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
