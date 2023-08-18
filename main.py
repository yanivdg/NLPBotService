from flask import Flask, request, jsonify
import logging
import nltk
import json
import requests
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

nltk.download('punkt')
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

def preprocess_text(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

@app.route('/answer', methods=['POST'])
def get_answer():
    try:
            #user_question = event["question"]
            data = request.get_json()
            user_question = data["question"]
            # Log a message
            #logging.info(f"Received question: {user_question}")
            response = requests.post('https://0ai42tfv4e.execute-api.us-west-1.amazonaws.com/default/MyNLPSearchService', json={"subject": user_question})
            #logging.info(response.json())
            response_data = response.json()  # Convert the response content to a dictionary
            # Extract the list of URLs from the "body" field
            body_list = json.loads(response_data['body'])
            # Print the list of URLs
            print(body_list)
            related_sites  = body_list 
            # Assuming you have fetched the related sites URLs
            #[
            #    "https://www.example.com/site1",
            #    "https://www.example.com/site2",
            # Add more URLs
            #]
            # Extract content from the related sites
            site_contents = []
            for url in related_sites:
                content = scrape_web_page(url)
                site_contents.append(content)
            # Assuming you have determined that the user's question is related to parrots
            if "parrot" in user_question.lower():
                logging.info(f"User question: {user_question}")
                best_answer = get_best_answer(user_question, site_contents)
	            # Log the best answer
                logging.info(f"Best answer: {best_answer}")
                return {
                    "statusCode": 200,
                    "body": json.dumps({"best_answer": best_answer})
                }
            else:
                return {
                    "statusCode": 200,
                    "body": json.dumps({"message": "Not a parrot-related question."})
                }
    except Exception as error:
            print("An error occurred:", error)
            return {
                "statusCode": error.args[0],
                "body": json.dumps(f"An error occurred:, {error}")
                }

def calculate_similarity(query, sentences):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    query_vector = tfidf_vectorizer.transform([query])
    similarity_scores = np.dot(query_vector, tfidf_matrix.T).toarray()[0]
    return similarity_scores

def get_best_answer(query, content_list):
    try:
        sentences = []
        for content in content_list:
            print(content)
            sentences.extend(preprocess_text(content))
        similarity_scores = calculate_similarity(query, sentences)
        most_similar_index = np.argmax(similarity_scores)
        return sentences[most_similar_index]
    except Exception as error:
                print("An error occurred:", error)
                return {
                    "statusCode": error.args[0],
                    "body": json.dumps(f"An error occurred:, {error}")
                    }

def scrape_web_page(url):
    # web scraping logic 
    response = requests.get(url)
    if response.status_code == 200:
        return response.text  # Return the HTML content of the page
    else:
        print("Failed to scrape content from " + url)
        return ''

#def lambda_handler(event, context):AWS Lambda only

if __name__ == '__main__':
    host = '0.0.0.0'
    port = 5000
    logging.info(f"Starting Flask app on {host}:{port}...")
    app.run(host=host, port=port)
