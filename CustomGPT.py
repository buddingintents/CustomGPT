import streamlit as st
import PyPDF2
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import string
import tensorflow as tf
import numpy as np
import os

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download
        
# Text preprocessing functions
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Lowercase
    text = text.lower()
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize and stem
    processed_tokens = []
    for token in filtered_tokens:
        lemma = lemmatizer.lemmatize(token)
        stem = stemmer.stem(lemma)
        processed_tokens.append(stem)
    return ' '.join(processed_tokens)

# Streamlit app
st.title("PDF Text Processing and Model Training")
# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    # Text extraction
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # Text preprocessing
    with st.spinner('Preprocessing text...'):
        processed_text = preprocess_text(text)
    
    # Model creation
    with st.spinner('Creating and training model...'):
        # Tokenize text
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts([processed_text])
        total_words = len(tokenizer.word_index) + 1
        
        # Create input sequences
        input_sequences = []
        for line in processed_text.split('\n'):
            token_list = tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)
        
        # Pad sequences
        max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(tf.keras.preprocessing.sequence.pad_sequences(
            input_sequences, maxlen=max_sequence_len, padding='pre'
        ))
        
        # Create predictors and labels
        X = input_sequences[:, :-1]
        y = input_sequences[:, -1]
        y = tf.keras.utils.to_categorical(y, num_classes=total_words)
        
        # Build model
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(total_words, 100, input_length=max_sequence_len-1),
            tf.keras.layers.LSTM(150),
            tf.keras.layers.Dense(total_words, activation='softmax')
        ])
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X, y, epochs=50, verbose=0, batch_size=32)
    
    # Model download
    model.save('text_model.h5')
    with open('text_model.h5', 'rb') as f:
        st.download_button(
            label="Download Trained Model",
            data=f,
            file_name='text_model.h5',
            mime='application/octet-stream'
        )
    
    # Usage instructions
    st.subheader("Usage Instructions")
    st.markdown("""
    1. Download the trained model file
    2. Install required dependencies:
       ```bash
       pip install tensorflow numpy nltk
       ```
    3. Use the sample code below for integration
    """)
    
    # Sample code
    st.subheader("Integration Sample Code (Python)")
    st.code("""
import tensorflow as tf
import numpy as np
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import string
# Load the trained model
model = tf.keras.models.load_model('text_model.h5')
# Define preprocessing functions (same as training)
# ... [include the preprocessing functions from above] ...
def generate_text(seed_text, next_words, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = tf.keras.preprocessing.sequence.pad_sequences(
            [token_list], maxlen=max_sequence_len-1, padding='pre'
        )
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text
# Example usage
preprocessed_text = preprocess_text("your input text here")
generated = generate_text(preprocessed_text, next_words=5, max_sequence_len=10)
print(generated)
    """)
st.markdown("---")
st.info("Note: This is a demonstration model. For production use, consider using larger datasets and more sophisticated architectures.")
