import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# loading the model and vectorizer
model = joblib.load(r'..\data\fake_news_classifer.pkl')
vectorizer = joblib.load(r'..\data\tfidf_vectorizer.pkl')

# streamlit app
st.title('Fake News Detection Application')

# taking title and article inputs
title = st.text_input('Enter the title of the news article')
article = st.text_input('Enter the content of the news article')

if st.button('Classify'):
    if title and article:
        input_data = title + ' ' + article
        input_vector = vectorizer.transform([input_data])
        predictions = model.predict(input_data)
        
        # result
        if predictions[0] == 'REAL':
            st.success('This news article is REAL.')
        if predictions[1] == 'FAKE':
            st.error('This article is FAKE')