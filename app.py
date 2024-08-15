from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import spacy
import pickle
import os
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Configure Gemini API
api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')

# Function to tokenize and lemmatize a text using spaCy
def tokenize_and_lemmatize(text):
    text = str(text)
    return [token.lemma_ for token in nlp(text) if not token.is_stop and token.is_alpha]

# Function to calculate the average vector for each document
def calculate_avg_vector(tokens, model, vector_size):
    vector_sum = sum(model.wv[word] for word in tokens if word in model.wv)
    if vector_sum.any():
        return vector_sum / len(tokens)
    else:
        return [0] * vector_size

def load_and_process_data():
    # Load the data
    df = pd.read_csv("./Data/Clusters (3).csv")
    
    # Process the data
   
    # Train Word2Vec model
    # word2vec_model = Word2Vec(
    #     sentences=df['SPACY_TOKENS'],
    #     vector_size=100,
    #     window=5,
    #     min_count=1,
    #     workers=4
    # )
    with open('./Model/word2vec_model.pkl', 'rb') as file:
        word2vec_model = pickle.load(file)
    
    # Calculate and store Word2Vec vectors
    df['WORD2VEC_VECTOR'] = df['SPACY_TOKENS'].apply(lambda tokens: calculate_avg_vector(tokens, word2vec_model, 100))
    
    # Stack the Word2Vec vectors into a matrix
    word2vec_matrix = pd.DataFrame(df['WORD2VEC_VECTOR'].tolist(), index=df.index)

    with open('./Model/kmeans_model.pkl', 'rb') as file:
        kmeans_word2vec_spacy = pickle.load(file)
    
    # Fit the model and predict clusters
    df['CLUSTER_WORD2VEC_SPACY'] = kmeans_word2vec_spacy.fit_predict(word2vec_matrix)

    return df, word2vec_model, kmeans_word2vec_spacy, word2vec_matrix

# Load data at startup
df, word2vec_model, kmeans_word2vec_spacy, word2vec_matrix = load_and_process_data()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query_text = request.form['question']
        
        # Gemini Integration
        prompt_template = f"""This is the question: {query_text} for the Customer Support Team, 
        so you need to give a great answer so that customer retention will be high, 
        think like you are working in customer support team from last 10 years
        so what answer will you give and don't write these things - As a Customer Support Engineer with over 10 years
        of experience, I would provide the following response, write like you are replying someone without 
        knowing them that how much experience do you have. Write answer directly"""

        response = model.generate_content(prompt_template)
        answer = response.text

        # Process query and find similar questions
        query_vector = calculate_avg_vector(tokenize_and_lemmatize(query_text), word2vec_model, 100)
        query_vector = np.array(query_vector).reshape(1, -1)
        
        # Predict the cluster for the new query
        query_cluster = kmeans_word2vec_spacy.predict(query_vector)[0]
        
        # Get indices of questions in the same cluster
        cluster_indices = np.where(df['CLUSTER_WORD2VEC_SPACY'] == query_cluster)[0]
        
        # Calculating cosine similarity with questions in the same cluster
        cosine_similarities = cosine_similarity(query_vector, word2vec_matrix.iloc[cluster_indices])
        
        # Get indices of the most similar questions
        most_similar_indices = cosine_similarities.argsort()[0, ::-1][:5]
        
        # Extract the most similar questions and answers
        similar_data = df.iloc[cluster_indices].iloc[most_similar_indices][['QUESTION_TEXT', 'ANSWER']]
        
        return jsonify({
            "answer": answer,
            "similar_questions": similar_data['QUESTION_TEXT'].tolist(),
            "query_cluster": int(query_cluster)
        })

    # For GET request, send initial suggestions
    initial_suggestions = df['QUESTION_TEXT'].head().tolist()
    return render_template('index.html', suggestions=initial_suggestions)

if __name__ == '__main__':
    app.run(debug=True, port = 9000)