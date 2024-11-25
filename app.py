import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import re

# Load the model
@st.cache_resource
def load_model_only():
    return load_model("transformer_model.h5")

model = load_model_only()

# Word index from training (example)
word_index = {
    "great": 1,
    "good": 2,
    "bad": 3,
    "terrible": 4,
    # Add more...
}

# Preprocess and tokenize text
def preprocess_and_tokenize(text, word_index, max_length=100):
    # Preprocessing
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Tokenize
    tokens = [word_index.get(word, 0) for word in text.split()]
    # Pad sequences
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    else:
        tokens += [0] * (max_length - len(tokens))
    return np.array([tokens])  # Add batch dimension

# Streamlit app interface
st.title("Sentiment Analysis with Transformer Model")

user_input = st.text_area("Enter a review to analyze:")

if st.button("Analyze"):
    if user_input.strip():
        input_data = preprocess_and_tokenize(user_input, word_index)
        prediction = model.predict(input_data)
        sentiment = ["Negative", "Neutral", "Positive"]
        st.write(f"**Predicted Sentiment:** {sentiment[np.argmax(prediction)]}")
    else:
        st.write("Please enter some text to analyze.")
