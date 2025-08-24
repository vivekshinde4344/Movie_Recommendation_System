import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(filename):
    df = pd.read_csv(filename)
    df.dropna(subset=['title', 'genres'], inplace=True)
    df['genres'] = df['genres'].str.replace('|', ' ', regex=False)
    return df


def build_genre_matrix(df):
    vectorizer = CountVectorizer()
    genre_matrix = vectorizer.fit_transform(df['genres'])
    return genre_matrix, vectorizer


def recommend_by_genre(input_genres, df, genre_matrix, vectorizer, top_n=5):
    
    input_genres = input_genres.replace('|', ' ')
    input_vec = vectorizer.transform([input_genres])
    
    similarities = cosine_similarity(input_vec, genre_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:top_n]
    
    return df.iloc[top_indices]['title'].tolist()


filename = "movie_dataset.csv"  
df = load_data(filename)
genre_matrix, vectorizer = build_genre_matrix(df)


user_genres = input("Enter your preferred genre(s) (e.g. Action or Action|Adventure|Sci-Fi): ")
recommendations = recommend_by_genre(user_genres, df, genre_matrix, vectorizer)

print(f"\nMovies that match your genre preference ({user_genres}):")
for movie in recommendations:
    print(f"- {movie}")
