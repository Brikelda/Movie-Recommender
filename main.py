import pandas as pd
import re
import numpy as np
import tkinter as tk
from tkinter import ttk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("ml-25m/movies.csv")
#print(movies)

def clean_title(title):
   title = re.sub("[^a-zA-Z0-9 ]", "",title)
   return title

movies["clean_title"] = movies["title"].apply(clean_title)

# print(movies)

vectorizer = TfidfVectorizer(ngram_range=(1,2))

tfidf = vectorizer.fit_transform(movies["clean_title"])
# Create Search function
def search(title):
   title = clean_title(title)
   query_vec = vectorizer.transform([title])
   similarity = cosine_similarity(query_vec,tfidf).flatten()
   indices = np.argpartition(similarity, -5)[-5:]
   results = movies.iloc[indices][::-1]
   return results

#otr
ratings = pd.read_csv("ml-25m/ratings.csv")

def find_similar_movies(movie_id):
# WE're gonna find the users who like the same movie as us and their recommendations (users similar to us)
   similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
# Their recommendations
   similar_users_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]

# What % of the users rated the movie >4, and take only the 10%
   similar_users_recs = similar_users_recs.value_counts() / len(similar_users)
   similar_users_recs = similar_users_recs[similar_users_recs > .1]

# all of the users who watched the movies recommended to us and rated them good
   all_users = ratings[(ratings["movieId"].isin(similar_users_recs.index)) & (ratings["rating"] > 4)]
# find the % that all users recommend each of these movies
   all_users_recs = all_users["movieId"].value_counts()/len(all_users["userId"].unique())

# What interests us is high % from users similar to us and a low % from all users
   rec_percentages = pd.concat([similar_users_recs,all_users_recs], axis=1)
   rec_percentages.columns = ["similar", "all"]
   rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
   rec_percentages = rec_percentages.sort_values("score", ascending=False)
# Take top10 rec
   return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]

# Define function to handle text input and display recommendations
def on_search():
    recommendation_tree.delete(*recommendation_tree.get_children())
    title = movie_name_input.get().strip()
    if len(title) > 5:
        results = search(title)
        movie_id = results.iloc[0]["movieId"]
        recommendations = find_similar_movies(movie_id)
        for index, row in recommendations.iterrows():
            recommendation_tree.insert("", tk.END, values=(row['title'], row['genres'], f"{row['score']:.2f}"))

root = tk.Tk()
root.title("Movie Recommender")

# create movie name input label and text box
movie_name_label = ttk.Label(root, text="Movie Title:")
movie_name_label.grid(column=0, row=0)
movie_name_input = ttk.Entry(root, width=30)
movie_name_input.grid(column=1, row=0)

# create search button
search_button = ttk.Button(root, text="Search", command=on_search)
search_button.grid(column=2, row=0)
# create recommendation treeview
recommendation_label = ttk.Label(root, text="Recommendations:")
recommendation_label.grid(column=0, row=1, sticky="w", padx=10, pady=10)

recommendation_tree = ttk.Treeview(root, columns=("title", "genre", "score"), show="headings")
recommendation_tree.heading("title", text="Title")
recommendation_tree.heading("genre", text="Genre")
recommendation_tree.heading("score", text="Score")
recommendation_tree.column("title", width=350)
recommendation_tree.column("genre", width=200)
recommendation_tree.column("score", width=100)
recommendation_tree.grid(column=0, row=2, columnspan=3, padx=10, pady=10)

def on_search():
    recommendation_tree.delete(*recommendation_tree.get_children()) # clear previous recommendations
    title = movie_name_input.get().strip()
    if len(title) > 5:
        results = search(title)
        movie_id = results.iloc[0]["movieId"]
        recommendations = find_similar_movies(movie_id)
    for index, row in recommendations.iterrows():
        recommendation_tree.insert("", tk.END, values=(row['title'], row['genres'], f"{row['score']:.2f}"))
# create search button
search_button = ttk.Button(root, text="Search", command=on_search)
search_button.grid(column=2, row=0, padx=10, pady=10)

root.mainloop()

