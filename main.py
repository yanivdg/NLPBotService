import json
from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

app = Flask(__name__)

# List of parrots
parrots = ["macaw", "cockatiel", "budgerigar", "lorikeet", "lovebird", "conure", "parakeet", "african grey", "amazon", "cockatoo", "eclectus", "pionus"]

# Web scraping function
def scrape_web_page(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            paragraphs = soup.find_all("p")
            data = " ".join([para.text for para in paragraphs])
            return data
    except requests.exceptions.RequestException:
        return None

# Preprocess the scraped data
def preprocess_text(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

# Process user query
def process_query(query):
    return preprocess_text(query)

# Check if the user query is related to parrots
def is_related_to_parrots(query):
    for q in query:
        if any(parrot in q.lower() for parrot in parrots):
            return True
    return False

# Calculate similarity
def calculate_similarity(query, sentences):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    query_vector = tfidf_vectorizer.transform([query])
    similarity_scores = np.dot(query_vector, tfidf_matrix.T).toarray()[0]
    return similarity_scores

# Get answer
def get_answer(query, sentences):
    similarity_scores = calculate_similarity(query, sentences)
    most_similar_index = np.argmax(similarity_scores)
    return sentences[most_similar_index]

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_query = data['query']

    processed_user_query = process_query(user_query)

    if is_related_to_parrots(processed_user_query):
        answer = get_answer(processed_user_query[0], sentences)
        return jsonify({"response": answer})
    else:
        return jsonify({"response": "Sorry, I'm here just to assist with parrots."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)