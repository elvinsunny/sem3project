import streamlit as st
import numpy as np
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the models, vectorizer, and label encoders
with open('category_classifier_model.pkl', 'rb') as model_file:
    model_category = pickle.load(model_file)

with open('maturity_level_classifier_model.pkl', 'rb') as model_file:
    model_maturity = pickle.load(model_file)

with open('context_classifier_model.pkl', 'rb') as model_file:
    model_context = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('label_encoder_category.pkl', 'rb') as encoder_file:
    label_encoder_category = pickle.load(encoder_file)

with open('label_encoder_maturity.pkl', 'rb') as encoder_file:
    label_encoder_maturity = pickle.load(encoder_file)

with open('label_encoder_context.pkl', 'rb') as encoder_file:
    label_encoder_context = pickle.load(encoder_file)

# Define the prediction function
def predict_joke_attributes(joke_text):
    # Clean the joke text
    joke_text_clean = re.sub(r'\s+', ' ', joke_text)
    joke_text_clean = re.sub(r'[^\w\s]', '', joke_text_clean)
    joke_text_clean = joke_text_clean.lower()

    # Transform text to feature vectors
    seq = vectorizer.transform([joke_text_clean])

    # Predict the category, maturity level, and context
    pred_category = model_category.predict(seq)
    pred_maturity = model_maturity.predict(seq)
    pred_context = model_context.predict(seq)

    category = label_encoder_category.inverse_transform(pred_category)[0]
    maturity_level = label_encoder_maturity.inverse_transform(pred_maturity)[0]
    context = label_encoder_context.inverse_transform(pred_context)[0]

    return category, maturity_level, context

# Streamlit interface
st.title('Joke Classification')
st.write('Enter a joke and predict its category, maturity level, and contextual usage.')

joke_text = st.text_area('Enter a joke:')
if st.button('Predict'):
    if joke_text:
        category, maturity_level, context = predict_joke_attributes(joke_text)
        st.write(f'The joke belongs to the category: {category}')
        st.write(f'The joke belongs to the maturity level: {maturity_level}')
        st.write(f'The joke is suitable for: {context}')
    else:
        st.write('Please enter a joke to predict.')
