import os
import json
import ast
import numpy as np
import pandas as pd
import random
import nltk
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Pastikan resource NLTK tersedia
nltk.download('punkt')
nltk.download('stopwords')

# Fungsi untuk mencari file dengan ekstensi .json, .csv, atau .parquet secara rekursif dengan menampilkan progres
def find_data_files(directory):
    # Hitung total file di direktori dan subdirektori
    total_files = 0
    for _, _, files in os.walk(directory):
        total_files += len(files)
        
    scanned_files = 0
    data_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            scanned_files += 1
            progress = int((scanned_files / total_files) * 100)
            if file.lower().endswith(('.json', '.csv', '.parquet')):
                file_path = os.path.join(root, file)
                data_files.append(file_path)
                print(f"{progress}%")
                print("Dir:", file_path)
    return data_files

# Fungsi untuk memuat data dari file JSON, CSV, atau Parquet
def load_data(file_path):
    if file_path.lower().endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    elif file_path.lower().endswith('.csv'):
        df = pd.read_csv(file_path)
        data = {"intents": []}
        for _, row in df.iterrows():
            tag = row['tag']
            patterns = row['patterns']
            if isinstance(patterns, str):
                if patterns.strip().startswith('['):
                    try:
                        patterns = ast.literal_eval(patterns)
                    except Exception:
                        patterns = patterns.split(';')
                else:
                    patterns = patterns.split(';')
            responses = row['responses']
            if isinstance(responses, str):
                if responses.strip().startswith('['):
                    try:
                        responses = ast.literal_eval(responses)
                    except Exception:
                        responses = responses.split(';')
                else:
                    responses = responses.split(';')
            data['intents'].append({
                'tag': tag,
                'patterns': patterns,
                'responses': responses
            })
    elif file_path.lower().endswith('.parquet'):
        df = pd.read_parquet(file_path)
        data = {"intents": []}
        for _, row in df.iterrows():
            tag = row['tag']
            patterns = row['patterns']
            if isinstance(patterns, str):
                if patterns.strip().startswith('['):
                    try:
                        patterns = ast.literal_eval(patterns)
                    except Exception:
                        patterns = patterns.split(';')
                else:
                    patterns = patterns.split(';')
            responses = row['responses']
            if isinstance(responses, str):
                if responses.strip().startswith('['):
                    try:
                        responses = ast.literal_eval(responses)
                    except Exception:
                        responses = responses.split(';')
                else:
                    responses = responses.split(';')
            data['intents'].append({
                'tag': tag,
                'patterns': patterns,
                'responses': responses
            })
    else:
        raise ValueError("Format file tidak didukung: " + file_path)
    return data

# Fungsi untuk memuat dan menggabungkan data dari semua file yang ditemukan
def load_all_data(directory):
    data_files = find_data_files(directory)
    if not data_files:
        raise ValueError("Tidak ada file JSON, CSV, atau Parquet yang ditemukan di directory: " + directory)
    combined_data = {"intents": []}
    for file in data_files:
        print("Loading file:", file)
        dataset = load_data(file)
        if "intents" in dataset:
            combined_data["intents"].extend(dataset["intents"])
        else:
            print("File", file, "tidak memiliki key 'intents'")
    return combined_data

# Fungsi untuk preprocessing data: tokenisasi dan pengumpulan kata
def preprocess_data(data):
    words, labels, docs_x, docs_y = [], [], [], []
    for intent in data['intents']:
        for pattern in intent['patterns']:
            tokens = word_tokenize(pattern)
            words.extend(tokens)
            docs_x.append(tokens)
            docs_y.append(intent['tag'])
        if intent['tag'] not in labels:
            labels.append(intent['tag'])
    words = sorted(set(words))
    labels = sorted(labels)
    return words, labels, docs_x, docs_y

# Konversi data ke format numerik (bag-of-words)
def prepare_training_data(words, labels, docs_x, docs_y):
    stop_words = set(stopwords.words('english'))
    training, output = [], []
    out_empty = [0] * len(labels)
    for i, doc in enumerate(docs_x):
        bag = [1 if w in doc and w not in stop_words else 0 for w in words]
        output_row = out_empty[:]
        output_row[labels.index(docs_y[i])] = 1
        training.append(bag)
        output.append(output_row)
    return np.array(training), np.array(output)

# Membuat model chatbot berbasis LSTM
def create_model(input_shape, output_shape):
    model = Sequential([
        LSTM(128, input_shape=(input_shape[1], 1), activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(output_shape, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

# Melatih model dan menyimpannya
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=200, batch_size=8, verbose=1)
    model.save("chatbot_model.h5")

# Memprediksi kelas (tag) untuk input pengguna
def predict_class(sentence, words, labels, model):
    tokens = word_tokenize(sentence)
    bag = np.array([1 if w in tokens else 0 for w in words]).reshape(1, -1, 1)
    res = model.predict(bag)
    return labels[np.argmax(res)]

# Fungsi interaksi chatbot
def chatbot(data, words, labels):
    model = tf.keras.models.load_model("chatbot_model.h5")
    print("AI Chatbot Ready! Ketik 'exit' untuk keluar.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        tag = predict_class(user_input, words, labels, model)
        responses = [i['responses'] for i in data['intents'] if i['tag'] == tag][0]
        print("Bot:", random.choice(responses))

if __name__ == "__main__":
    # Ubah data_directory sesuai kebutuhan; contoh di sini adalah direktori root '/'
    data_directory = "/"  
    data = load_all_data(data_directory)
    words, labels, docs_x, docs_y = preprocess_data(data)
    X_train, y_train = prepare_training_data(words, labels, docs_x, docs_y)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    
    # Membuat dan melatih model
    model = create_model(X_train.shape, len(labels))
    train_model(model, X_train, y_train)
    
    # Mulai interaksi dengan chatbot
    chatbot(data, words, labels)
