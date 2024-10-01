import streamlit as st
import pickle
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load NLTK stopwords and lemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocess function to clean and tokenize input text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(f"[{string.punctuation}]", " ", text)
    # Tokenize words
    tokens = text.split()
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Load the trained Naive Bayes model and CountVectorizer
def load_model():
    with open('sentiments_model.pkl', 'rb') as model_file:
        model, vectorizer = pickle.load(model_file)
    return model, vectorizer

# Main function to run the Streamlit app
def main():
    st.title("Sentiment Classifier: Happy or Sad")
    st.write("This app classifies review sentiments into either 'happy' or 'sad'.")

    # Input text for classification
    review_input = st.text_area("Enter a review:", placeholder="Type your review here...")

    if st.button("Classify"):
        if review_input:
            # Preprocess the input text
            preprocessed_text = preprocess_text(review_input)
            # Load model and vectorizer
            model, vectorizer = load_model()
            # Transform input using the CountVectorizer loaded with the model
            transformed_input = vectorizer.transform([preprocessed_text])
            # Predict sentiment
            prediction = model.predict(transformed_input)

            # Display the result
            if prediction == 1:
                st.success("Sentiment: Happy 😊")
            else:
                st.error("Sentiment: Sad 😢")
        else:
            st.warning("Please enter a review for classification.")

if __name__ == '__main__':
    main()
