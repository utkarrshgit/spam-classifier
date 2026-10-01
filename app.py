import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("models/spam_classifier.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

st.title("Spam Classifier")

message = st.text_area("Enter a message:")

if st.button("Classify"):
    if message.strip():
        message_vec = vectorizer.transform([message])
        prediction = model.predict(message_vec)[0]

        if prediction == 1:
            st.error("SPAM")
        else:
            st.success("HAM")
    else:
        st.warning("Please enter a message.")