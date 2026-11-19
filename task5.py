import streamlit as st
import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


st.set_page_config(
    page_title="Movie Recommendation System"
)

# st.title("Movie Recommendation System")
st.markdown(
    """
    <h1 style='color:#FF5733; text-align:center;'>
        🎬 Movie Recommendation Systemstream
    </h1>
    """,
    unsafe_allow_html=True
)
st.write("Select a Movie to get Similar Movie Recommendation")

@st.cache_data
def load_data():
    data = pd.read_csv("cleaned_movies.csv")
    data["overview"] = data["overview"].fillna("")
    data = data.dropna(subset=["title"])
    data = data.drop_duplicates(
        subset=["title"]
    )
    data = data.reset_index(drop=True)
    return data

data = load_data()
@st.cache_data
def preprocessing(text):

    stop_words = set(ENGLISH_STOP_WORDS)
    def clean_text(value):
        value = value.lower()
        value = re.sub(r"[^a-zA-Z\s]","",value)
        value = re.sub(r"\s+"," ",value).strip()

        words = value.split()

        words = [
            word for word in words
            if word not in stop_words
        ]
        return " ".join(words)

    return text.apply(clean_text)

data["clean_text"] = preprocessing(
    data["overview"]
)

text = data["clean_text"].fillna("").astype(str)

@st.cache_data
def create_vectors(text):
    tfidf = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1,2)
    )

    tfidf_matrix = tfidf.fit_transform(text)
    return tfidf_matrix

tfidf_matrix = create_vectors(text)

similarity_matrix = cosine_similarity(
    tfidf_matrix
)

def recommend(item_names,top_n=5):

    movie_index = data[
        data["title"].str.lower() == item_names.lower()
    ].index

    if len(movie_index) == 0:
        return None

    movie_index = movie_index[0]

    scores = list(
        enumerate(similarity_matrix[movie_index])
        )

    scores = sorted(
        scores,
        key = lambda x:x[1],
        reverse=True
    )

    scores=scores[1:top_n+1]
    movie_indices = [i[0] for i in scores]
 

    recommendations = data.iloc[movie_indices][
        ["title"]
    ].copy()

    recommendations["Similarity Score"] = [
        round(i[1],3) for i in scores
    ]

    return recommendations

movie_list = data["title"].tolist()

selected_movie = st.selectbox(
    "Select a Movie",
    movie_list
)

if st.button("Generate Recommendations"):
    results = recommend(selected_movie,5)

    if results is not None and not results.empty:
        st.subheader("Recommended Movies")

        for index, row in results.iterrows():

            st.write(
                f"{row['title']}",
                f" - Similarity:{row['Similarity Score']}"
            )
    else:
        st.warning("No Recommendations Found")





