import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_data
def load_data():

    movies = pd.read_csv("data/tmdb_5000_movies.csv")
    credits = pd.read_csv("data/tmdb_5000_credits.csv")

    movies = movies.merge(
        credits,
        left_on="id",
        right_on="movie_id"
    )

    def parse_names(value):
        if pd.isna(value):
            return []

        data = ast.literal_eval(value)
        return [item["name"] for item in data]

    def parse_cast(value):
        if pd.isna(value):
            return []

        data = ast.literal_eval(value)
        return [item["name"] for item in data[:3]]

    def get_director(value):
        if pd.isna(value):
            return []

        data = ast.literal_eval(value)

        for item in data:
            if item["job"] == "Director":
                return [item["name"]]

        return []

    movies["genres"] = movies["genres"].apply(parse_names)
    movies["keywords"] = movies["keywords"].apply(parse_names)
    movies["cast"] = movies["cast"].apply(parse_cast)
    movies["crew"] = movies["crew"].apply(get_director)

    movies["overview"] = movies["overview"].fillna("")

    movies["tags"] = (
        movies["overview"] + " " +
        movies["genres"].apply(lambda x: " ".join(x)) + " " +
        movies["keywords"].apply(lambda x: " ".join(x)) + " " +
        movies["cast"].apply(lambda x: " ".join(x)) + " " +
        movies["crew"].apply(lambda x: " ".join(x))
    )

    return movies


movies = load_data()

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["tags"])

similarity = cosine_similarity(tfidf_matrix)


def recommend(movie):

    index = movies[
        movies["title_x"].str.lower() == movie.lower()
    ].index[0]

    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    return [
        movies.iloc[i[0]]["title_x"]
        for i in movie_list
    ]


st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Recommendation System")

st.write(
    "Find movies similar to your favorite movie using Machine Learning."
)

movie_list = movies["title_x"].sort_values().unique()

selected_movie = st.selectbox(
    "Select a movie:",
    movie_list
)

if st.button("Recommend Movies 🎯"):

    recommendations = recommend(selected_movie)

    st.subheader("Recommended Movies 🍿")

    for i, movie in enumerate(recommendations, start=1):
        st.write(f"{i}. {movie}")