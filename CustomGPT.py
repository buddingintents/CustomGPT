import streamlit as st 
import tensorflow as tf
import numpy as np
import PyPDF2
import re
import nltk
import os
import json
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

nltk.data.path.append('/tmp/nltk_data')
nltk.download('punkt', download_dir='/tmp/nltk_data')
nltk.download('wordnet', download_dir='/tmp/nltk_data')

# Preprocessing Functions
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    return text

def lemmatize_and_stem(text):
    lemmatizer = nltk.WordNetLemmatizer()
    stemmer = PorterStemmer()
    words = word_tokenize(text)
    processed_words = [stemmer.stem(lemmatizer.lemmatize(word, wordnet.VERB)) for word in words]
    return " ".join(processed_words)

# Streamlit UI
st.title("PDF Text Processing & TensorFlow Model Trainer")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from PDF
    extracted_text = extract_text_from_pdf(uploaded_file)
    cleaned_text = clean_text(extracted_text)
    processed_text = lemmatize_and_stem(cleaned_text)
    
    st.subheader("Extracted and Processed Text:")
    st.text_area("Processed Text", processed_text, height=200)
    
    # Tokenization
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([processed_text])
    sequences = tokenizer.texts_to_sequences([processed_text])
    vocab_size = len(tokenizer.word_index) + 1

    # Padding
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post")

    # Model Creation
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=50, input_length=max_length),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(50, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    # Train the model on dummy labels (just for demonstration)
    dummy_labels = np.random.randint(2, size=(1,))
    model.fit(padded_sequences, dummy_labels, epochs=5, verbose=1)

    # Save the model
    model_path = "text_model.h5"
    model.save(model_path)

    # Provide Download Link
    with open(model_path, "rb") as f:
        st.download_button("Download Trained Model", f, file_name="text_model.h5")

    # Display Integration Instructions
    st.subheader("Model Integration Instructions")

    sample_code = """
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # Load the model
    model = tf.keras.models.load_model("text_model.h5")

    def preprocess_text(input_text):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts([input_text])
        sequence = tokenizer.texts_to_sequences([input_text])
        padded = pad_sequences(sequence, maxlen=100, padding="post")
        return padded

    input_text = "Sample query"
    processed_input = preprocess_text(input_text)

    prediction = model.predict(processed_input)
    print("Model Output:", prediction)
    """

    st.code(sample_code, language="python")
