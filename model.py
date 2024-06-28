from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Load the CSV file
df = pd.read_csv('AllDataset2.csv')

# Verify columns
assert 'Problem' in df.columns and 'Solution' in df.columns, "CSV file must contain 'Problem' and 'Solution' columns"

questions = df['Problem'].tolist()
answers = df['Solution'].tolist()

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the questions
question_embeddings = model.encode(questions)


def find_most_similar_question(user_question, question_embeddings, questions):
    # Generate embedding for the user question
    user_question_embedding = model.encode([user_question])[0]

    # Calculate cosine similarities
    similarities = cosine_similarity([user_question_embedding], question_embeddings)[0]

    # Find the index of the most similar question
    most_similar_index = np.argmax(similarities)

    return questions[most_similar_index], similarities[most_similar_index]



def get_answer(user_question):
    most_similar_question, similarity = find_most_similar_question(user_question, question_embeddings, questions)
    if similarity > 0.7:  # Threshold to ensure the question is relevant
        answer_index = questions.index(most_similar_question)
        return answers[answer_index]
    else:
        return "Sorry, I don't understand the question. Can you please rephrase?"

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    user_question = data.get('question')
    if not user_question:
        return jsonify({'error': 'Question is required'}), 400

    answer = get_answer(user_question)
    return jsonify({'question': user_question, 'answer': answer})

if __name__ == '__main__':
    app.run(debug=True, port=5555)
