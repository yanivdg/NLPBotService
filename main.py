import requests
from flask import Flask, request
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

app = Flask(__name__)

# List of parrots
parrots = ["parrot","parrot","macaw", "cockatiel", "budgerigar", "lorikeet", "lovebird", "conure", "parakeet", "african grey", "amazon", "cockatoo", "eclectus", "pionus"]

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
    # Your preprocessing steps here (tokenization, stemming, etc.)
    # For simplicity, let's just split the text into sentences.
    sentences = nltk.sent_tokenize(text)
    return sentences

# Process user query
def process_query(query):
    return preprocess_text(query)

def calculate_similarity(query, sentences):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    query_vector = tfidf_vectorizer.transform([query])
    similarity_scores = np.dot(query_vector, tfidf_matrix.T).toarray()[0]  # Convert to NumPy array and access the first (and only) row
    return similarity_scores

# Check if the user query is related to parrots
def is_related_to_parrots(query):
    for q in query:
        if any(parrot in q.lower() for parrot in parrots):
            return True
    return False

def get_answer(query, sentences):
    similarity_scores = calculate_similarity(query, sentences)
    most_similar_index = np.argmax(similarity_scores)
    return sentences[most_similar_index]

@app.route('/parrot', methods=['POST'])
def parrot_handler():
    data = request.get_json()
    user_query = data['user_query']

    # Web scraping example from Wikipedia
    url = "https://en.wikipedia.org/wiki/Parrot"
    scraped_data = scrape_web_page(url)

    if scraped_data is None:
        return {
            'response': "Error: Unable to fetch data from the web."
        }

    # Preprocess the scraped data
    sentences = preprocess_text(scraped_data)

    # Process user query
    processed_user_query = process_query(user_query)

    # Check if the user query is related to parrots
    if is_related_to_parrots(processed_user_query):
        # Get the most relevant answer about parrots
        answer = get_answer(processed_user_query[0], sentences)
        return {
            'response': answer
        }
    else:
        return {
            'response': "Sorry, I'm here just to assist with parrots."
        }

if __name__ == '__main__':
    app.run(debug=False)
