import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'short_dataset.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
    # episodes_df = pd.DataFrame(data['episodes'])
    # reviews_df = pd.DataFrame(data['reviews'])
    titles = data['titles']
    reviews = data['reviews']
    titles_df = pd.DataFrame({"titles": titles, "reviews": reviews})


app = Flask(__name__)
CORS(app)

# Sample search using json with pandas
def json_search(query):
    matches = []
    merged_df = pd.merge(episodes_df, reviews_df, left_on='id', right_on='id', how='inner')
    matches = merged_df[merged_df['title'].str.lower().str.contains(query.lower())]
    matches_filtered = matches[['title', 'descr', 'imdb_rating']]
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json

# return similar albums based on title
def title_search(query):
    matches = []
    # merged_df = pd.merge(episodes_df, reviews_df, left_on='id', right_on='id', how='inner')
    # matches = merged_df[merged_df['title'].str.lower().str.contains(query.lower())]
    matches_filtered = matches[['title']]
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json

#return similar classical titles based on input title query
def find_similar_titles(query, dataset):
    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(dataset['titles'])

    query_vec = vectorizer.transform([query])

    cosine_scores = cosine_similarity(query_vec, tfidf_matrix)

    top_indices = cosine_scores.argsort()[0][:][::-1]

    top_titles = pd.DataFrame({"albums": [dataset.loc[i]['titles'] for i in top_indices], "reviews": [dataset.loc[i]['reviews'] for i in top_indices]})
    return top_titles


def jaccard_similarity(set1, set2):
    intersection = len(set(set1).intersection(set2))
    union = len(set(set1).union(set2))
    return intersection / union if union != 0 else 0

def find_most_relevant_albums(input_album, album_themes_map):
    if input_album not in album_themes_map:
        print(f"Album '{input_album}' not found.")
        return []
    input_themes = album_themes_map[input_album]
    similarity_scores = []
    for album, themes in album_themes_map.items():
        if album == input_album:
            continue
        score = jaccard_similarity(input_themes, themes)
        similarity_scores.append((album, score))
    sorted_albums = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    return sorted_albums


# this what we working on rn
def combine_rankings(dataset, title_input, genre_input, composer_input, top_n=8):
    ''' 
    get a list of rankings from each algorithm, then will weight avg to 
    get final result
    '''
    top_titles = find_similar_titles(title_input, dataset)

    if genre_input != None:
        pass #TODO

    if composer_input != None:
        pass #TODO

    output = top_titles[:top_n]
    top_json = output.to_json(orient='records')
    return top_json

# routes
@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    return json_search(text)

@app.route("/albums")
def albums_search():
    text = request.args.get("title")
    genre = request.args.get("genre")
    composer = request.args.get("composer")
    return combine_rankings(titles_df, text, genre, composer)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)