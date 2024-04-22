import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import numpy as np

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
    artists = data['Artist'].values()
    reviews = data['Review'].values()
    titles = data['Album'].values()
    eras = data['Era'].values()
    titles_df = pd.DataFrame({"titles": titles, "artists": artists, "reviews": reviews, "eras": eras})

title_reverse_index = {t: i for i, t in enumerate(titles_df["titles"])}
composer_reverse_index = {t: i for i, t in enumerate(titles_df["artists"])}

app = Flask(__name__)
CORS(app)


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

    tfidf_matrix = vectorizer.fit_transform(dataset['titles'].fillna(""))

    query_vec = vectorizer.transform([query])

    cosine_scores = cosine_similarity(query_vec, tfidf_matrix)

    top_indices = cosine_scores.argsort()[0][:][::-1]
    
    top_titles = pd.DataFrame({"titles": [dataset.loc[i]['titles'] for i in top_indices], 
                               "reviews": [dataset.loc[i]['reviews'] for i in top_indices]})
    return top_titles


def jaccard_similarity(set1, set2):
    intersection = len(set(set1).intersection(set2))
    union = len(set(set1).union(set2))
    return intersection / union if union != 0 else 0

def inverted_index(album_name, dataset):
    for index, name in enumerate(dataset["titles"]):
        if album_name != name:
            continue
        else:
            return index

def theme_similarity_scores(input_album, dataset):
    location = inverted_index(input_album, dataset)
    input_themes = dataset["themes"][location]
    similarity_scores = []
    for themes in dataset["themes"]:
        if input_themes == themes:
            continue
        else:
            score = jaccard_similarity(input_themes, themes)
            similarity_scores.append(score)
    return similarity_scores

# SVD similarity
def SVD(input_album, df):
    albums = df["titles"].to_list()
    reviews = df["reviews"].fillna("")
    # Step 2: Preprocess and Vectorize the Text Data
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    X = vectorizer.fit_transform(reviews)

    # Step 3: Apply SVD to Reduce Dimensionality
    svd_model = TruncatedSVD(n_components=100)  # Adjust the number of components as needed
    latent_matrix = svd_model.fit_transform(X)

    # Step 4: Compute Similarities Between the Reviews
    similarity_matrix = cosine_similarity(latent_matrix)

    if input_album not in albums:
        print("Album not found.")
        return []

    # Get the index of the input album
    index = albums.index(input_album)

    # Create a list of similarity scores with other albums
    similarity_scores = []
    for i in range(len(similarity_matrix)):
        if albums[i] != input_album:  # Ignore the input album itself
            similarity_scores.append((albums[i], similarity_matrix[index][i]))

    # Sort the albums based on similarity score in descending order
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # Get the top similar albums
    output = pd.DataFrame({"title": [s[0] for s in similarity_scores], "scores": [s[1] for s in similarity_scores]})
    return output

# Update similarity scores after SVD by filtering eras
def era_filter(composer_input, df): # the df here is the output df after applying SVD; titles_df is the global variable for the original dataset's dataframe
  era_list = {"Medieval": 0, "Renaissance": 1, "Baroque": 2, "Classical": 3, "Early Romantic": 4, "Late Romantic": 5, "20th and 21st century": 6}
  composer_index = composer_reverse_index[composer_input]
  composer_era = titles_df["Era"][composer_index]
  composer_era_num = era_list[composer_era]
  df_era_num_list = []
  df_era_diff_list = []
  for n in range(len(df)):
    title = df.iloc[n]["titles"]
    title_index = title_reverse_index[title]
    title_era = titles_df["Era"][title_index]
    title_era_num = era_list[title_era]
    df_era_num_list.append(title_era_num)
    diff = title_era_num - composer_era_num
    df_era_diff_list.append(diff)
  df["era"] = df_era_num_list
  df["era_diff"] = df_era_diff_list
  df["updated_score"] = .7 * df["scores"] + .3 * (1 / (df["era_diff"] + 1))
  output = df.sort_values(by='updated_score', ascending=False)
  return output



# this what we working on rn
def combine_rankings(dataset, title_input, composer_input, purpose_input, top_n=8):
    ''' 
    get a list of rankings from each algorithm, then will weight avg to 
    get final result
    '''
    
    top_titles = find_similar_titles(title_input, dataset) #TODO delete when we have dropdown

    top_title = top_titles.iloc[0]

    title_svd_output = SVD(top_title["titles"], titles_df)
    output = title_svd_output[:top_n]

    # filter by composer era
    if composer_input != "":
        era_output = era_filter(composer_input, title_svd_output)
        output = era_output[:top_n] # TODO LATER 

    if purpose_input != "":
        pass #TODO
    
    # weight all of the rankings, get a list of titles sorted properly

    # for each i in top_n, get the title in the ranked list (output), then use
    # title_reverse_index to get the proper row from titles_df
    # make a list of the series, then we'll convert it to a df
    titles = []
    composers = []
    eras = []
    scores = []
    # descriptions = [] 
    # title, composer, era, score
    for i in range(top_n):
        title = output["title"].iloc[i]
        score = output["scores"].iloc[i]
        row = (dataset.loc[dataset['titles'] == title])

        if not row.empty:
            row = row.iloc[0]
            titles.append(row["titles"])
            composers.append(row["artists"])
            eras.append(row["eras"])
            scores.append(score)



    new_output = pd.DataFrame({"title": titles, "artists": composers, "era": eras, "scores": score})

    # top_json becomes ranked dataframe to json file

    
    top_json = new_output.to_json(orient='records')
    return top_json

# routes
@app.route("/")
def home():
    return render_template('base.html',title="sample html")


@app.route("/albums")
def albums_search():
    text = request.args.get("title")
    genre = request.args.get("genre")
    composer = request.args.get("composer")
    return combine_rankings(titles_df, text, genre, composer)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)