import pandas as pd
import nltk
import streamlit as st
import praw
from dotenv import load_dotenv
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Load environment variables from .env file
load_dotenv()

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Access the Reddit credentials from environment variables
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Reddit API Setup using PRAW with credentials from .env
reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                     client_secret=REDDIT_CLIENT_SECRET,
                     user_agent=REDDIT_USER_AGENT)

# Function to extract text from Reddit post
def extract_text_from_reddit(url):
    try:
        submission = reddit.submission(url=url)
        submission_text = submission.title + "\n\n" + submission.selftext
        return submission_text
    except Exception as e:
        return f"Error retrieving content from Reddit: {e}"

# Load dataset (sample data preparation)
data = pd.read_csv(r"train.csv", encoding='ISO-8859-1')

# Preprocess text data
data['text'] = data['text'].fillna('').astype(str)
data['text'] = data['text'].apply(preprocess_text)

# Encode sentiment labels
label_encoder = LabelEncoder()
data['sentiment'] = label_encoder.fit_transform(data['sentiment'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['sentiment'], test_size=0.2, random_state=42)

# Build pipeline with TF-IDF Vectorizer and Naive Bayes classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('classifier', MultinomialNB())
])

# Train the model
pipeline.fit(X_train, y_train)

# Function to predict sentiment
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    prediction = pipeline.predict([processed_text])
    return label_encoder.inverse_transform(prediction)[0]

# Streamlit UI
st.title("Sentiment Analysis App")

# Dropdown for choosing between text or Reddit link input
input_choice = st.radio("Choose an option:", ('Enter text', 'Enter Reddit URL'))

if input_choice == 'Enter text':
    # Text input
    user_input = st.text_area("Enter a movie review or social media post:")
elif input_choice == 'Enter Reddit URL':
    # URL input for Reddit
    reddit_url = st.text_input("Enter the Reddit URL of the review:")

# Predict button
if st.button("Predict Sentiment"):
    if input_choice == 'Enter text':
        if user_input:
            sentiment = predict_sentiment(user_input)
            st.write(f"Predicted sentiment: {sentiment}")
        else:
            st.write("Please enter some text to analyze.")
    elif input_choice == 'Enter Reddit URL':
        if reddit_url:
            extracted_text = extract_text_from_reddit(reddit_url)
            if "Error" not in extracted_text:
                st.write(f"Extracted Text (first 1000 characters):\n{extracted_text[:1000]}")
                sentiment = predict_sentiment(extracted_text)
                st.write(f"Predicted sentiment: {sentiment}")
            else:
                st.write(extracted_text)  # Display error if Reddit URL extraction fails
        else:
            st.write("Please enter a valid Reddit URL.")
