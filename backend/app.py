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
from sklearn.preprocessing import normalize
import numpy as np
from numpy.linalg import svd
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
import random

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
    artists = data['composer'].values()
    reviews = data['review'].values()
    titles = data['title'].values()
    eras = data['era'].values()
    short_review = data['short_review'].values()
    
    joy = data['joy'].values()
    sadness = data['sadness'].values()
    fear = data['fear'].values()
    anger = data['anger'].values()
    neutral = data['neutral'].values()
    emotions_df = pd.DataFrame({ "joy":joy, "sadness":sadness, "fear":fear, "anger":anger, "neutral":neutral})
    titles_df = pd.DataFrame({"title": titles, "composer": artists, "review": reviews, "short_review": short_review, "era": eras})

title_reverse_index = {t: i for i, t in enumerate(titles_df["title"])}
composer_reverse_index = {t: i for i, t in enumerate(titles_df["composer"])}
global_title = None

app = Flask(__name__)
CORS(app)

#########################################################################################
# find similarity between user input and albums for first step of search
def first_step(query, dataset, n=5):
    corpus = []
    for _, row in dataset.iterrows():
        album = row['title'] if not pd.isnull(row['title']) else ""
        review = row['review'] if not pd.isnull(row['review']) else ""
        artist = row['composer'] if not pd.isnull(row['composer']) else ""
        era = row['era'] if not pd.isnull(row['era']) else ""
        row_data = ' '.join([album, review, artist, era])
        corpus.append(row_data)

    print("made corpus")

    # from svd_demo-kickstarter-2024-inclass.ipynb
    vectorizer = TfidfVectorizer(stop_words = 'english', max_df = .7, min_df = 75)
    td_matrix = vectorizer.fit_transform(corpus)
    docs_compressed, s, words_compressed = svds(td_matrix, k=50)
    words_compressed = words_compressed.transpose()

    # word_to_index = vectorizer.vocabulary_
    # index_to_word = {i:t for t,i in word_to_index.items()}
    # words_compressed_normed = normalize(words_compressed, axis = 1)
    docs_compressed_normed = normalize(docs_compressed)
    print("compressed")

    query_tfidf = vectorizer.transform([query]).toarray()
    query_vec = normalize(np.dot(query_tfidf, words_compressed)).squeeze()

    sims = docs_compressed_normed.dot(query_vec)
    asort = np.argsort(-sims)[:n]


    top_titles = pd.DataFrame({
        "title": dataset.loc[asort]['title'].values,
        "composer": dataset.loc[asort]['composer'].values,
    })
    # print(top_titles)

    return top_titles.to_json(orient='records')
        

def title_to_link(s):
    if s is None:
        s = ""
    return "+".join(s.split())

def get_title_series(titles):
    return titles.apply(lambda s : title_to_link(s))


####################################################################################################
# return similar albums based on title
def title_search(query):
    matches = []
    matches_filtered = matches[['title']]
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json


def find_similar_title(query, dataset):
    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(dataset['title'].fillna(""))

    query_vec = vectorizer.transform([query])

    cosine_scores = cosine_similarity(query_vec, tfidf_matrix)

    top_indices = cosine_scores.argsort()[0][:][::-1]
    
    top_titles = [dataset.loc[i]['title'] for i in top_indices[:5]]
    return top_titles


def jaccard_similarity(set1, set2):
    intersection = len(set(set1).intersection(set2))
    union = len(set(set1).union(set2))
    return intersection / union if union != 0 else 0

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

def get_reviews(query, dataset):
    # Retrieve the index of the title from the dataset
    index = title_reverse_index.get(query)
    if index is not None:
        # Return the review at the found index
        print(index)
        return dataset.iloc[index]['review']
    return ""  # Return empty string if title not found

# find similarity between albums based on emotion:
def do_svd(mat, k=0, option=False):
    U, Sigma, VT = svd(mat)
    U = pd.DataFrame(U[:, :k])
    VT = pd.DataFrame(VT[:k, :])
    if option:
        return Sigma
    else:
        return U, VT
####################################################################################################
# Essentailly Andy 2nd Step Search Part #
def find_similar_reviews(query, dataset):
    # Vectorize reviews
    vectorizer_reviews = TfidfVectorizer()
    tfidf_matrix_reviews = vectorizer_reviews.fit_transform(dataset['review'].fillna(""))

    # Get the review for the query using the corrected function call
    query_review = get_reviews(query, titles_df)
    query_vec_reviews = vectorizer_reviews.transform([query_review])

    # Calculate cosine similarity between the query and the titles dataset
    cosine_scores_reviews = cosine_similarity(query_vec_reviews, tfidf_matrix_reviews).flatten()

    # Normalize scores
    max_reviews_score = np.max(cosine_scores_reviews) if np.max(cosine_scores_reviews) != 0 else 1
    normalized_reviews_scores = cosine_scores_reviews / max_reviews_score

    # Sort indices based on normalized reviews scores
    top_indices = normalized_reviews_scores.argsort()[::-1]
    print(top_indices)

    # Assign ranks based on the sorting order
    ranks = range(1, len(top_indices) + 1)

    # Create DataFrame with the combined top results
    top_titles = pd.DataFrame({
        "title": dataset.iloc[top_indices]['title'].values,
        "composer": dataset.iloc[top_indices]['composer'].values,
        "short_review": dataset.iloc[top_indices]['short_review'].values,
        "era": dataset.iloc[top_indices]['era'].values,
        "rank": ranks,  # Include ranks instead of scores
        "link": get_title_series(dataset.iloc[top_indices]['title'])
    })

    return top_titles


def cos_sim_album(title_input, dataset):
    if title_input not in title_reverse_index:
        print("Title not found in the index.")
        return pd.DataFrame()  # Return an empty DataFrame if title not found

    album_index = title_reverse_index[title_input]
    
    vectorizer = TfidfVectorizer()
    
    # Ensure that 'review' column is not empty and contains string data
    if dataset['review'].isnull().any():
        dataset['review'] = dataset['review'].fillna('')  # Fill NA/NaN values with empty string

    tfidf_matrix = vectorizer.fit_transform(dataset['review'])
    
    # Transform the review of the album at the given index into the same TF-IDF space
    query_vec = vectorizer.transform([dataset.iloc[album_index]['review']])
    
    # Compute cosine similarity between the query vector and all review vectors
    cosine_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get the indices of the scores sorted by similarity (highest first)
    top_indices = np.argsort(-cosine_scores)
    
    # Fetch data for the top indices - limit the number of results if necessary
    top_titles = pd.DataFrame({
        "title": dataset.iloc[top_indices]['title'].values,
        "composer": dataset.iloc[top_indices]['composer'].values,
        "short_review": dataset.iloc[top_indices]['short_review'].values,
        "era": dataset.iloc[top_indices]['era'].values,
        "score": cosine_scores[top_indices],  # Include cosine similarity scores
        "link": get_title_series(dataset.iloc[top_indices]['title'])
    })
    
    return top_titles

def find_similar_composers(query, dataset, same_composer):
    # Normalize the input and create a copy of the dataset for manipulation
    artists_normalized = dataset['composer'].str.lower()
    query_lower = query.lower()
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(artists_normalized.fillna(""))
    query_vec = vectorizer.transform([query_lower])
    
    cosine_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_index = cosine_scores.argmax()
    input_artist_era = dataset.iloc[top_index]['era']
    updated_artist = dataset.iloc[top_index]['composer']
    if same_composer:
        top_composers = dataset[dataset['composer'].str.lower() == updated_artist]
    else:
        top_composers = dataset[(dataset['era'] == input_artist_era) & (dataset['composer'].str.lower() != updated_artist)]
    
    return top_composers


def find_similar_album_by_emotion(emotions_df, titles_df, query_index):
    # Check if the query_index is valid
    if query_index >= len(titles_df):
        print(f"Query index {query_index} is out of bounds for titles DataFrame.")
        return pd.DataFrame()  # Return an empty DataFrame if the index is out of bounds

    # Apply SVD to the emotion scores
    U, _ = do_svd(emotions_df, k=5)

    # Compute cosine similarity
    query_vector = U.iloc[query_index, :].values.reshape(1, -1)
    similarities = cosine_similarity(query_vector, U).flatten()

    # Exclude the query row itself and get top indices
    top_indices = np.argsort(-similarities)
    top_indices = top_indices[top_indices != query_index]

    # Safe indexing
    if not np.all(top_indices < len(titles_df)):
        print("One or more indices are out of bounds after filtering.")
        top_indices = [i for i in top_indices if i < len(titles_df)]

    # Create DataFrame with the top results including normalized scores, title, composer, and review
    top_results = pd.DataFrame({
        "title": titles_df.iloc[top_indices]['title'].values,
        "composer": titles_df.iloc[top_indices]['composer'].values,
        "short_review": titles_df.iloc[top_indices]['short_review'].values,
        "era": titles_df.iloc[top_indices]['era'].values,
        "rank": range(1, len(top_indices) + 1),  # Generate ranks for the filtered indices
        "link": get_title_series(titles_df.iloc[top_indices]['title'])
    })

    return top_results


####################################################################################################
def combine_rankings(emotions_df, titles_df, title_input, composer_input, same_composer, top_n=10):
    # Fetch similar composers first as a filtering step
    composer_filtered_results = find_similar_composers(composer_input, titles_df, same_composer)


    # Run similar reviews based on filtered dataset
    review_results = find_similar_reviews(title_input, composer_filtered_results)
    review_results = review_results.rename(columns={'composer': 'composer_review', 'short_review': 'short_review_review', 'era': 'era_review'})
    review_results['review_rank'] = review_results['rank']  # assuming 'rank' comes from the find_similar_reviews output

    # Run emotion analysis based on filtered dataset
    title_index = title_reverse_index.get(title_input, -1)
    if title_index == -1:
        print("Title not found for emotion analysis.")
        return pd.DataFrame()

    emotion_results = find_similar_album_by_emotion(emotions_df, composer_filtered_results, title_index)
    emotion_results = emotion_results.rename(columns={'composer': 'composer_emotion', 'short_review': 'short_review_emotion', 'era': 'era_emotion'})
    emotion_results['emotion_rank'] = emotion_results['rank']  # assuming 'rank' comes from the find_similar_album_by_emotion output

    # Merge the two results on 'title'
    merged_results = pd.merge(review_results, emotion_results, on='title', suffixes=('_review', '_emotion'))

    # Calculate combined rank (you can adjust the weights here)
    merged_results['combined_rank'] = merged_results['review_rank'] + merged_results['emotion_rank']
    final_results = merged_results.dropna()
    final_results = final_results.drop_duplicates(subset=['title'])

    final_results.columns = final_results.columns.str.replace('_review', '')

    # Sort by combined rank again
    final_results = final_results.sort_values(by='combined_rank').head(top_n)

    # Convert to JSON
    final_json = final_results[['title', 'composer', 'short', 'era', 'review_rank', 'emotion_rank', 'combined_rank', 'link']].to_json(orient='records')

    return final_json

# Usage example, ensure all functions and variables are defined and loaded appropriately
# final_output = combine_rankings(dataset, emotions_df, titles_df, 'Some Title', 'Some Composer')
# print(final_output)

####################################################################################################

# routes
@app.route("/")
def home():
   return render_template('base.html',title="sample html")

@app.route("/input")
def get_first_step():
   query = request.args.get("text")
   return first_step(query, titles_df)

@app.route("/albums")
def albums_search():
    text = global_title
    composer = request.args.get("composer")
    exclusion = False if request.args.get("exclude") != "null" else True
    print(exclusion)
    # purpose = request.args.get("composer")
    return combine_rankings(emotions_df, titles_df, text, composer, exclusion)

# function for multiple pages from 
# https://stackoverflow.com/questions/67351167/one-flask-with-multiple-page
@app.route('/page_two')
def page_two():
   return render_template('page_two.html')

@app.route('/home')
def go_home():
   return render_template('base.html')

@app.route('/store_title', methods=["POST"])
def store_title():
   print("storing title...")
   global global_title 
   title = request.json.get("title_input")
   global_title = title
   print("title stored")
   print(global_title)
   return render_template('page_two.html')

@app.route('/get_title')
def get_title():
   return json.dumps(global_title)

if 'DB_NAME' not in os.environ:
   app.run(debug=True,host="0.0.0.0",port=5000)


# def test_combined_rankings():
#     title_input = "Sonatas and Rondos"
#     composer_input = "Andy Li"
#     same_compoers = False
#     top_n = 10

#     # Assuming the combined_rankings function is properly defined and ready to use
#     result_json = combine_rankings(emotions_df, titles_df, title_input, composer_input, same_compoers, top_n)
    
#     # Print the combined rankings result in a formatted way
#     formatted_json = json.dumps(json.loads(result_json), indent=4)  # Pretty print the JSON
#     print("Combined Rankings JSON Output:")
#     print(formatted_json)
# # Run the test
# test_combined_rankings()

