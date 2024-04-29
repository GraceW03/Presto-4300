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
from scipy.spatial.distance import cdist

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
    
titles_df = pd.DataFrame({"title": titles, "composer": artists, "review": reviews, "era": eras, "short_review" : short_review})

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

    # corpus = dataset["titles"].str.cat([dataset['reviews'], dataset['artists'], 
    #                                     dataset['eras']], sep =" ").to_list()

    print("made corpus")

    # from svd_demo-kickstarter-2024-inclass.ipynb
    vectorizer = TfidfVectorizer(stop_words = 'english', max_df = .7, min_df = 50)
    td_matrix = vectorizer.fit_transform(corpus)
    docs_compressed, s, words_compressed = svds(td_matrix, k=100)
    words_compressed = words_compressed.transpose()
    # word_to_index = vectorizer.vocabulary_
    # index_to_word = {i:t for t,i in word_to_index.items()}
    # words_compressed_normed = normalize(words_compressed, axis = 1)
    docs_compressed_normed = normalize(docs_compressed)
    print("compressed")

    query_tfidf = vectorizer.transform([query]).toarray()
    query_vec = normalize(np.dot(query_tfidf, words_compressed)).squeeze()

    sims = docs_compressed_normed.dot(query_vec)
    asort = np.argsort(-sims)[:n+1]


    top_titles = pd.DataFrame({
        "title": dataset.loc[asort]['title'].values,
        "composer": dataset.loc[asort]['composer'].values,
    })
    # print(top_titles)

    return top_titles.to_json(orient='records')
        
    






# demo code for similarity function
# def closest_projects_to_query(query_vec_in, k = 5):
#     sims = docs_compressed_normed.dot(query_vec_in)
#     asort = np.argsort(-sims)[:k+1]
#     return [(i, documents.iloc(i)["Album"],sims[i]) for i in asort[1:]]





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

####################################################################################################
### helper function
def get_reviews(query, dataset):
    index = title_reverse_index[query]
    index_of_reviews_column = dataset.columns.get_loc('review')
    album_row = dataset.iloc[index]
    return album_row[index_of_reviews_column]

#getting short reviews for explainbility in output
def get_short_reviews(query, dataset):
    index = title_reverse_index[query]
    index_of_short_reviews_column = dataset.columns.get_loc('short_review')
    album_row = dataset.iloc[index]
    return album_row[index_of_short_reviews_column]


def find_similar_reviews(query, dataset):
    # Vectorize reviews
    vectorizer_reviews = TfidfVectorizer()
    tfidf_matrix_reviews = vectorizer_reviews.fit_transform(dataset['review'].fillna(""))

    # Transform the query to fit the trained TF-IDF model for reviews
    query_vec_reviews = vectorizer_reviews.transform([get_reviews[query, dataset]])

    # Calculate cosine similarity between the query and the titles dataset
    cosine_scores_reviews = cosine_similarity(query_vec_reviews, tfidf_matrix_reviews).flatten()
    max_reviews_score = np.max(cosine_scores_reviews) if np.max(cosine_scores_reviews) != 0 else 1
    normalized_reviews_scores = cosine_scores_reviews / max_reviews_score

    # Sort indices based on normalized reviews scores
    top_indices = normalized_reviews_scores.argsort()[::-1]

    # Create DataFrame with the combined top results
    top_titles = pd.DataFrame({
        "title": dataset.iloc[top_indices]['title'].values,
        "composer": dataset.iloc[top_indices]['composer'].values,
        "review": dataset.iloc[top_indices]['review'].values,
        "score": normalized_reviews_scores[top_indices]  # Include normalized reviews scores for reference
    })

    return top_titles

# does the same thing hmmm
def cos_sim_album(title_input, dataset):
    album_index = title_reverse_index[title_input]

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(dataset['review'])

    query_vec = vectorizer.transform([dataset.iloc[album_index]['review']])

    cosine_scores = cosine_similarity(query_vec, tfidf_matrix)

    top_indices = cosine_scores.argsort()[0][:][::-1]

    top_titles = pd.DataFrame({
        "title": dataset.loc[top_indices]['title'].values,
        "composer": dataset.loc[top_indices]['composer'].values,
        "review": dataset.loc[top_indices]['review'].values,
        "score": cosine_scores[top_indices]  # Include final scores for reference
    })
    return top_titles

def find_similar_composers(query, dataset):
    # Normalize the input and create a copy of the dataset for manipulation
    artists_normalized = dataset['composer'].str.lower()

    # Vectorize the artist names
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(artists_normalized.fillna(""))

    # Transform the query to fit the trained TF-IDF model
    query_vec = vectorizer.transform([query.lower()])

    # Calculate cosine similarity between the query and the dataset
    cosine_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Determine the era of the input artist based on the highest cosine similarity
    top_index = cosine_scores.argmax()
    input_artist_era = dataset.iloc[top_index]['era']

    # Calculate era scores (1 if matching era, 0 otherwise)
    era_scores = (dataset['era'] == input_artist_era).astype(int)

    # Normalize scores to ensure fair weighting
    normalized_cosine_scores = cosine_scores / cosine_scores.max()
    normalized_era_scores = era_scores  # Already 0 or 1

    # Combine the scores with 50% weight each
    final_scores = 0.5 * normalized_cosine_scores + 0.5 * normalized_era_scores
    reranked_indices = final_scores.argsort()[::-1]

    # Create a DataFrame of the top similar artists after reranking
    top_composers = pd.DataFrame({
        "title": dataset.loc[reranked_indices]['title'].values,
        "composer": dataset.loc[reranked_indices]['composer'].values,
        "review": dataset.loc[reranked_indices]['review'].values,
        "score": final_scores[reranked_indices]  # Include final scores for reference
    })

    return top_composers

def combine_review_and_composer_search(title_query, composer_query, dataset):
    # Retrieve similar titles based on the title query
    similar_reviews = find_similar_reviews(title_query, dataset)
    
    # Retrieve similar composers based on the composer query
    similar_composers = find_similar_composers(composer_query, dataset)
    
    # Merge the results on common titles, averaging the scores
    if not similar_reviews.empty and not similar_composers.empty:
        combined_results = pd.merge(similar_reviews, similar_composers, on=['title', 'composer', 'review'], suffixes=('_review', '_composer'))
        combined_results['score'] = (combined_results['score_review'] + combined_results['score_composer']) / 2
    else:
        return pd.DataFrame()  # Return empty if any of the searches yield no results

    # Sort by the combined score in descending order
    combined_results = combined_results.sort_values(by='score', ascending=False)

    # Return the top N results
    return combined_results[['title', 'composer', 'review', 'score']]


####################################################################################################
def combine_rankings(dataset, title_input, composer_input, top_n=10):
    ''' 
    Get a list of rankings from each algorithm, then will weight avg to 
    get final result
    '''
    if composer_input and title_input:
        top_titles_artists = combine_review_and_composer_search(title_input, composer_input, dataset)
        output = top_titles_artists.iloc[:top_n]

    elif title_input:
        top_titles = find_similar_reviews(title_input, dataset)
        output = top_titles.iloc[:top_n]

    elif composer_input:
        top_composers = find_similar_composers(composer_input, dataset)
        output = top_composers.iloc[:top_n]

    # have to add explainbility by showing common words/first two lines of reviews

    # Extracting relevant information for output
    titles = output["title"].tolist()
    artists = output["composer"].tolist()
    eras = output["era"].tolist()
    scores = output["score"].tolist()

    # Create DataFrame with categorized scores
    new_output = pd.DataFrame({
        "title": titles,
        "composer": artists,
        "score": scores  # Use the categorized similarity labels
    })

    # Convert DataFrame to JSON
    top_json = new_output.to_json(orient='records')
    return top_json
####################################################################################################

# routes
@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/input")
def get_first_step():
   print("here1")
   query = request.args.get("text")
   return first_step(query, titles_df)


@app.route("/albums")
def albums_search():
    text = request.args.get("title")
    composer = request.args.get("composer")
    # purpose = request.args.get("composer")
    return combine_rankings(titles_df, text, composer)

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