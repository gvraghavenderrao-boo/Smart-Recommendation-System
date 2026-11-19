# Movie Recommendation System

## Project Description

This project is a Movie Recommendation System developed using Python and Streamlit.

It recommends similar movies based on the movie selected by the user. It uses Natural Language Processing (NLP), TF-IDF Vectorization and Cosine Similarity to identify movies with similar descriptions.

The application provides a simple interface where users can select a movie and get the top 5 similar movie recommendations.

## Technologies Used

1. Python
2. Streamlit
3. Pandas
4. Regular Expressions (re)
5. Scikit-learn
6. TF-IDF Vectorizer
7. Cosine Similarity

## Project Workflow

1. Load cleaned movie dataset
2. Handle missing values and duplicate records
3. Text Preprocessing
4. Remove special characters and stop words
5. TF-IDF Vectorization
6. Calculate Cosine Similarity
7. Select a movie
8. Generate Top 5 Movie Recommendations
9. Display results using Streamlit

## Project Implementation

### 1. Load Dataset

- Read the `cleaned_movies.csv` file.
- Handle missing movie descriptions.
- Remove records with missing movie titles.
- Remove duplicate movie titles.

### 2. Text Preprocessing

- Convert text into lowercase.
- Remove special characters and numbers.
- Remove extra spaces.
- Remove English stop words.

### 3. TF-IDF Vectorization

- Convert cleaned movie descriptions into numerical vectors.
- Use `TfidfVectorizer`.
- Consider unigrams and bigrams.
- Limit the maximum features to 3000.

### 4. Cosine Similarity

- Calculate similarity between movie descriptions.
- Generate a similarity matrix.
- Identify movies with similar content.

### 5. Movie Recommendation

- Select a movie from the dropdown list.
- Calculate similarity scores.
- Sort movies based on similarity.
- Display the top 5 similar movies.

### 6. Streamlit Application

- Display the Movie Recommendation System title.
- Provide a movie selection dropdown.
- Generate recommendations using a button.
- Display recommended movie titles and similarity scores.

## How to Run the Project

### Step 1: Install Required Libraries

```bash
pip install streamlit pandas scikit-learn
```

### Step 2: Run the Application

```bash
streamlit run app.py
```

### Step 3: Open the Application

Streamlit will open the Movie Recommendation System in your web browser.

Select a movie and click the **Generate Recommendations** button to view the top 5 similar movies.

## Project Files

1. `app.py` – Main Python application.
2. `cleaned_movies.csv` – Movie dataset used for recommendations.
3. `requirements.txt` – Required Python libraries.

## Output

The application displays:

- Selected movie.
- Top 5 recommended movies.
- Similarity score for each recommended movie.

## Conclusion

This project demonstrates how Natural Language Processing and Machine Learning techniques can be used to build a content-based Movie Recommendation System.

TF-IDF converts movie descriptions into numerical vectors, and Cosine Similarity identifies movies with similar content. Streamlit provides an interactive interface for generating movie recommendations.