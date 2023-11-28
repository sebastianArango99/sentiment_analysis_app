
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
from keras.models import load_model
import pickle
import matplotlib.pyplot as plt
import joblib

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, TFAutoModel
import nltk 

import requests
from io import BytesIO

@st.cache
def load_model_from_github(model_url, model_dir='./modelo_bert/'):
    # Create the directory if it doesn't exist
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # Download the model file
    response = requests.get(model_url)
    if response.status_code == 200:
        with open(os.path.join(model_dir, 'model.pb'), 'wb') as file:
            file.write(response.content)
    
    # Load the model
    loaded_model = tf.saved_model.load(model_dir)
    return loaded_model

checkpoint = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)


tok = Tokenizer(num_words=4000)


# Function to plot comparison bar chart
def plot_comparison(score_1, score_2):
    fig, ax = plt.subplots()
    ax.bar(['Modelo 1-BERT', 'Modelo 2-LSTM'], [score_1, score_2], color=['blue', 'green'])
    ax.set_ylabel('Score Modelo')
    ax.set_title('Comparación de Análisis de Sentimiento')
    return fig

# Load your models
#model_1 = joblib.load("C:/Users/Sebastián/OneDrive/Desktop/sentiment_model.joblib")


# Function to load the model
def load_model(model_directory):
    # Load the TensorFlow SavedModel
    model_2 = tf.keras.models.load_model(model_directory)
    return model_2

model_2 = load_model("./best_model.h5")#tf.keras.models.load_model("C:/models/modelo_lstm")
#model_1= tf.keras.models.load_model("./tiny_bert/")#load_model_from_github('https://github.com/sebastianArango99/sentiment_analysis_app/blob/main/modelo_bert/saved_model.pb')#tf.saved_model.load("./modelo_bert/")
#serving_default = model_1.signatures['serving_default']

def tokenization(data, **kwargs):
    return tokenizer(data,
                   padding=kwargs.get('padding','longest'),
                   max_length=kwargs.get('max_length',55),
                   truncation=True,
                   return_tensors="tf")

from transformers import BertTokenizer

# Load the BERT tokenizer
tokenizer =  BertTokenizer.from_pretrained("prajjwal1/bert-tiny")

# Load the tokenizer

with open('./tokenizer.pickle', 'rb') as handle:
    tok = pickle.load(handle)


def prepare_input(input_text, tokenizer, maxlen):
    # Tokenize the text to create a sequence of integers
    sequences = tokenizer.texts_to_sequences([input_text])
    # Pad sequences to have the same length
    padded_sequences = pad_sequences(sequences, maxlen=500)
    return padded_sequences

def predict(input_text):
    inputs = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=55,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='tf'
    )
    # Dummy token_type_ids
    token_type_ids = tf.zeros_like(inputs['input_ids'])

    prediction = model.predict([inputs['input_ids'], 
                                inputs['attention_mask'], 
                                token_type_ids])
    return prediction

# Streamlit app title
st.title('Comparación de Análisis de Sentimiento')

# User input
user_input = st.text_area("Introduce el texto a analizar aquí:", "")

input_text = user_input
encoded_input = tokenizer.encode_plus(
    input_text,
    add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
    max_length=4000,  # Max length to truncate/pad
    padding='max_length',  # Pad sentence to max length
    return_attention_mask=True,  # Return attention mask
    return_tensors='tf',  # Return TensorFlow tensors
)

ps = nltk.PorterStemmer()
def stemming_on_text(data):
    text = [ps.stem(word) for word in data]
    return data

user_input= ps.stem(user_input)
nltk.download('wordnet')

lm = nltk.WordNetLemmatizer()

user_input= lm.lemmatize(user_input) 
if st.button('Analizar Fragmento'):
   
    if user_input:
        # Predict using both models
        prepared_input = prepare_input(user_input, tok, 1000)
        encoded_input = tokenizer.encode_plus(input_text, max_length=55, 
                                      pad_to_max_length=True, 
                                      return_tensors="tf")

        # Construct the input in the format expected by the model
        model_inputs = {
            'input_3': encoded_input['input_ids'],  # assuming 'input_3' corresponds to 'input_ids'
            'input_4': encoded_input['attention_mask']  # assuming 'input_4' corresponds to 'attention_mask'
        }

        # Make the prediction
        #prediction_1 = serving_default(**model_inputs)
        
        #prediction_1 =predict(input_text)# serving_default(**tokenizer.encode_plus(input_text, return_tensors="tf"))
        prediction_2 = model_2.predict(prepared_input)

        # Extract scores from the predictions
        #score_1 = prediction_1['dense_3'].numpy()[0, 0]
        score_2 = prediction_2

        # Display results
        #label1 = 'Positivo' if score_1 > 0.5 else 'Negativo'
        label2 = 'Positivo' if score_2 > 0.5 else 'Negativo'
        #st.write('Resultado del Modelo 1 (BERT): ', label1)
        #st.write('Resultado del Modelo 1 (Score-Modelo): ', score_1)
        st.write('Resultado del Modelo 2 (LSTM): ', label2)
        st.write('Resultado del Modelo 2 (Score-Modelo): ', score_2[0,0])

        # Plot and display comparison chart
        #comparison_chart = plot_comparison(score_1, score_2)
        #st.pyplot(comparison_chart)
    else:
        st.write('Por favor, introduce algún texto para analizar.')


