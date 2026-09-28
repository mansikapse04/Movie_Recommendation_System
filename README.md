
# 🎬 Movie Recommendation System

A Machine Learning based Movie Recommendation System that recommends similar movies based on user preferences.

## 🚀 Features

- Select a movie from the available movie dataset
- Get 5 similar movie recommendations
- Content-Based Recommendation
- TF-IDF Vectorization
- Cosine Similarity
- Interactive Streamlit Web App

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Jupyter Notebook

## 🧠 Machine Learning Approach

This project uses Content-Based Filtering.

Movie information such as:

- Overview
- Genres
- Keywords
- Cast
- Director

is combined into a single `tags` feature.

TF-IDF is then used to convert the text data into numerical vectors.

Cosine Similarity is used to calculate similarity between movies.

## 📂 Project Structure

Movie_Recommendation_System/
│
├── data/
├── notebooks/
│   └── dataset_check.ipynb
│
├── src/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
````

### 2. Create virtual environment

```bash
python -m venv venv
```

### 3. Activate environment

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Add the movie dataset

Place:

* `tmdb_5000_movies.csv`
* `tmdb_5000_credits.csv`

inside the `data/` folder.

### 6. Run the application

```bash
streamlit run app.py
```

## 📌 Project Type

Machine Learning | Recommendation System | NLP | Streamlit

## 👩‍💻 Author

Mansi Kapse
