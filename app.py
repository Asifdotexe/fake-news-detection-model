import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Assuming you have a trained model and TF-IDF vectorizer
df = pd.read_csv('news.csv')  # Replace with your actual file path
labels = df['label']
text = df['text']

# Splitting the data into training and testing set
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

# Initializing the vectorizer and fit-transform it to the training data
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)

# Transform the test data using the same vectorizer
tfidf_test = tfidf_vectorizer.transform(x_test)

# Initializing the model and fitting it
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Streamlit application
st.title("Fake News Detection")

# User input
user_input = st.text_area("Enter news text")

if st.button("Predict"):
    if user_input:
        # Transform the input text using the same vectorizer
        user_input_transformed = tfidf_vectorizer.transform([user_input])

        # Predict using the trained model
        prediction = pac.predict(user_input_transformed)

        # Display the result
        result = "Fake" if prediction[0] == 0 else "Real"
        st.write(f'The news is: {result}')
    else:
        st.write("Please enter the news text.")
