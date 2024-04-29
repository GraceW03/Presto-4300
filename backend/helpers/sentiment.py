import pandas as pd
from transformers import pipeline
from tqdm.auto import tqdm
tqdm.pandas()

def preprocess_text(text):
    return text.lower()

emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", truncation=True)

def apply_emotion_analysis(text):
    try:
        results = emotion_pipeline(text)
        dominant_emotion = max(results, key=lambda x: x['score'])
        return {"emotion": dominant_emotion['label'].upper(), "score": dominant_emotion['score']}
    except Exception as e:
        print(f"Error in emotion analysis: {e}")
        return {"emotion": "ERROR", "score": 0}

def recommend_albums_by_emotion(df, emotion, top_n=10):
    filtered_df = df[df['dominant_emotion'] == emotion.upper()] if emotion else df
    top_albums = filtered_df.sort_values(by='emotion_score', ascending=False).head(top_n)
    return top_albums[['album_title', 'artists', 'dominant_emotion', 'emotion_score']]