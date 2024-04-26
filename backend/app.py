import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds
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

#########################################################################################
# find similarity between user input and albums for first step of search
def first_step(dataset):
    vectorizer = TfidfVectorizer()
    td_matrix = vectorizer.fit_transform(dataset)
    # from svd_demo-kickstarter-2024-inclass.ipynb
    u, s, v_trans = svds(td_matrix)
    docs_compressed, s, words_compressed = svds(td_matrix, k=40)
    words_compressed = words_compressed.transpose()


    






####################################################################################################
# return similar albums based on title
def title_search(query):
    matches = []
    matches_filtered = matches[['title']]
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json

#return similar classical titles based on input title query
@app.route("/get_titles", methods=['GET'])
def get_titles():
    user_input = request.args.get('query','')
    if user_input:
        try:
            filtered_names = find_similar_titles(user_input, titles_df)
            print(filtered_names)
            return jsonify(filtered_names)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify([])


def find_similar_titles(query, dataset):
    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(dataset['titles'].fillna(""))

    query_vec = vectorizer.transform([query])

    cosine_scores = cosine_similarity(query_vec, tfidf_matrix)

    top_indices = cosine_scores.argsort()[0][:][::-1]
    
    top_titles = [dataset.loc[i]['titles'] for i in top_indices[:5]]
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

def categorize_similarity(scores):
    # Define categories based on score ranges
    categories = []
    for score in scores:
        if score > 0.8:
            categories.append("Perfect Match")
        elif score > 0.6:
            categories.append("Very Similar")
        elif score > 0.4:
            categories.append("Similar")
        else:
            categories.append("Less Similarity")
    return categories

####################################################################################################
def find_similar_titles(query, dataset):
    # Step 1: Vectorize titles
    vectorizer_titles = TfidfVectorizer()
    tfidf_matrix_titles = vectorizer_titles.fit_transform(dataset['titles'].fillna(""))

    # Transform the query to fit the trained TF-IDF model for titles
    query_vec_titles = vectorizer_titles.transform([query])

    # Calculate cosine similarity between the query and the titles dataset
    cosine_scores_titles = cosine_similarity(query_vec_titles, tfidf_matrix_titles).flatten()
    max_title_score = np.max(cosine_scores_titles) if np.max(cosine_scores_titles) != 0 else 1
    normalized_titles_scores = cosine_scores_titles / max_title_score

    # Step 2: Apply SVD on reviews to capture semantic similarities
    vectorizer_reviews = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix_reviews = vectorizer_reviews.fit_transform(dataset['reviews'].fillna(""))

    # Apply SVD
    svd_model = TruncatedSVD(n_components=min(1000, tfidf_matrix_reviews.shape[1] - 1))
    latent_matrix_reviews = svd_model.fit_transform(tfidf_matrix_reviews)

    # Compute cosine similarities using the low-dimensional space created by SVD
    svd_query_review = svd_model.transform(vectorizer_reviews.transform([query]))
    cosine_scores_reviews = cosine_similarity(svd_query_review, latent_matrix_reviews).flatten()
    max_review_score = np.max(cosine_scores_reviews) if np.max(cosine_scores_reviews) != 0 else 1
    normalized_reviews_scores = cosine_scores_reviews / max_review_score

    # Step 3: Combine normalized scores (50% each from titles and reviews)
    combined_scores = 0.5 * normalized_titles_scores + 0.5 * normalized_reviews_scores

    # Sort indices based on combined scores
    top_indices = combined_scores.argsort()[::-1]

    # Step 4: Create DataFrame with the combined top results
    top_titles = pd.DataFrame({
        "titles": dataset.iloc[top_indices]['titles'].values,
        "artists": dataset.iloc[top_indices]['artists'].values,
        "reviews": dataset.iloc[top_indices]['reviews'].values,
        "scores": combined_scores[top_indices]  # Include combined scores for reference
    })

    return top_titles

def find_similar_composers(query, dataset):
    # Normalize the input and create a copy of the dataset for manipulation
    artists_normalized = dataset['artists'].str.lower()

    # Vectorize the artist names
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(artists_normalized.fillna(""))

    # Transform the query to fit the trained TF-IDF model
    query_vec = vectorizer.transform([query.lower()])

    # Calculate cosine similarity between the query and the dataset
    cosine_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Determine the era of the input artist based on the highest cosine similarity
    top_index = cosine_scores.argmax()
    input_artist_era = dataset.iloc[top_index]['eras']

    # Calculate era scores (1 if matching era, 0 otherwise)
    era_scores = (dataset['eras'] == input_artist_era).astype(int)

    # Normalize scores to ensure fair weighting
    normalized_cosine_scores = cosine_scores / cosine_scores.max()
    normalized_era_scores = era_scores  # Already 0 or 1

    # Combine the scores with 50% weight each
    final_scores = 0.5 * normalized_cosine_scores + 0.5 * normalized_era_scores
    reranked_indices = final_scores.argsort()[::-1]

    # Create a DataFrame of the top similar artists after reranking
    top_composers = pd.DataFrame({
        "titles": dataset.loc[reranked_indices]['titles'].values,
        "artists": dataset.loc[reranked_indices]['artists'].values,
        "reviews": dataset.loc[reranked_indices]['reviews'].values,
        "scores": final_scores[reranked_indices]  # Include final scores for reference
    })

    return top_composers

def combine_title_and_composer_search(title_query, composer_query, dataset):
    # Retrieve similar titles based on the title query
    similar_titles = find_similar_titles(title_query, dataset)
    
    # Retrieve similar composers based on the composer query
    similar_composers = find_similar_composers(composer_query, dataset)
    
    # Merge the results on common titles, averaging the scores
    if not similar_titles.empty and not similar_composers.empty:
        combined_results = pd.merge(similar_titles, similar_composers, on=['titles', 'artists', 'reviews'], suffixes=('_title', '_composer'))
        combined_results['scores'] = (combined_results['scores_title'] + combined_results['scores_composer']) / 2
    else:
        return pd.DataFrame()  # Return empty if any of the searches yield no results

    # Sort by the combined score in descending order
    combined_results = combined_results.sort_values(by='scores', ascending=False)

    # Return the top N results
    return combined_results[['titles', 'artists', 'reviews', 'scores']]


####################################################################################################
def combine_rankings(dataset, title_input, composer_input, purpose_input, top_n=10):
    ''' 
    Get a list of rankings from each algorithm, then will weight avg to 
    get final result
    '''
    if composer_input and title_input:
        top_titles_artists = combine_title_and_composer_search(title_input, composer_input, dataset)
        output = top_titles_artists.iloc[:top_n]

    elif title_input:
        top_titles = find_similar_titles(title_input, dataset)
        output = top_titles.iloc[:top_n]

    elif composer_input:
        top_composers = find_similar_composers(composer_input, dataset)
        output = top_composers.iloc[:top_n]

    if purpose_input != "Select a purpose (optional)":
        pass #TODO

    # Extracting relevant information for output
    titles = output["titles"].tolist()
    artists = output["artists"].tolist()
    scores = output["scores"].tolist()

    # Create DataFrame with categorized scores
    new_output = pd.DataFrame({
        "title": titles,
        "artists": artists,
        "scores": scores  # Use the categorized similarity labels
    })

    # Convert DataFrame to JSON
    top_json = new_output.to_json(orient='records')
    return top_json
####################################################################################################

# routes
@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/album_names", methods=["GET"])
def get_albums():
    return jsonify(titles_df["titles"].to_list())


@app.route("/albums")
def albums_search():
    text = request.args.get("title")
    composer = request.args.get("composer")
    return combine_rankings(titles_df, text, composer)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)