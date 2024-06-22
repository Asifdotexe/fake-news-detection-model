import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# loading the model and vectorizer
model = joblib.load(r'..\data\fake_news_classifer.pkl')
vectorizer = joblib.load(r'..\data\tfidf_vectorizer.pkl')