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
