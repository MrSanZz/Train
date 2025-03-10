import os
import sys
import json
import ast
import re
import sqlite3
import numpy as np
import pandas as pd
import random
import nltk
import tensorflow as tf
import xml.etree.ElementTree as ET
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Pastikan resource NLTK tersedia
nltk.download('punkt')
nltk.download('stopwords')

# Fungsi untuk auto-detect kolom untuk tag, patterns, responses
def auto_detect_columns(df):
    col_tag = None
    col_patterns = None
    col_responses = None

    # Cari berdasarkan nama kolom (case-insensitive)
    for col in df.columns:
        col_lower = col.lower()
        if "tag" in col_lower and col_tag is None:
            col_tag = col
        elif "pattern" in col_lower and col_patterns is None:
            col_patterns = col
        elif "response" in col_lower and col_responses is None:
            col_responses = col

    # Jika belum terdeteksi dan jumlah kolom tepat 3, gunakan urutan kolom
    if (col_tag is None or col_patterns is None or col_responses is None) and len(df.columns) == 3:
        cols = df.columns.tolist()
        col_tag, col_patterns, col_responses = cols[0], cols[1], cols[2]

    if col_tag is None or col_patterns is None or col_responses is None:
        raise ValueError("Tidak dapat mendeteksi kolom yang diperlukan secara otomatis.")
    return col_tag, col_patterns, col_responses

# Fungsi untuk mencari file dengan ekstensi yang diinginkan secara rekursif dengan progress menggunakan \r
def find_data_files(directory):
    valid_exts = ('.json', '.csv', '.parquet', '.xml', '.txt', '.log', '.sql', '.db', '.npy')
    total_files = sum(len(files) for _, _, files in os.walk(directory))
    scanned_files = 0
    data_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            scanned_files += 1
            progress = int((scanned_files / total_files) * 100)
            file_path = os.path.join(root, file)
            if file.lower().endswith(valid_exts):
                data_files.append(file_path)
                sys.stdout.write(f"\r{progress}% Dir: {file_path}")
                sys.stdout.flush()
    print()  # pindah ke baris baru setelah selesai
    return data_files

# Fungsi untuk memuat data dari berbagai format file
def load_data(file_path):
    if file_path.lower().endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    elif file_path.lower().endswith('.csv'):
        df = pd.read_csv(file_path)
        try:
            col_tag, col_patterns, col_responses = auto_detect_columns(df)
        except Exception as e:
            raise ValueError(f"CSV file {file_path} tidak dapat dideteksi kolomnya: {e}")
        data = {"intents": []}
        for _, row in df.iterrows():
            tag = row[col_tag]
            patterns = row[col_patterns]
            responses = row[col_responses]
            # Proses kolom patterns
            if isinstance(patterns, str):
                if patterns.strip().startswith('['):
                    try:
                        patterns = ast.literal_eval(patterns)
                    except Exception:
                        patterns = patterns.split(';')
                else:
                    patterns = patterns.split(';')
            # Proses kolom responses
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
        try:
            col_tag, col_patterns, col_responses = auto_detect_columns(df)
        except Exception as e:
            raise ValueError(f"Parquet file {file_path} tidak dapat mendeteksi kolomnya: {e}")
        data = {"intents": []}
        for _, row in df.iterrows():
            tag = row[col_tag]
            patterns = row[col_patterns]
            responses = row[col_responses]
            if isinstance(patterns, str):
                if patterns.strip().startswith('['):
                    try:
                        patterns = ast.literal_eval(patterns)
                    except Exception:
                        patterns = patterns.split(';')
                else:
                    patterns = patterns.split(';')
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
    elif file_path.lower().endswith('.xml'):
        tree = ET.parse(file_path)
        root = tree.getroot()
        data = {"intents": []}
        for intent in root.findall('intent'):
            tag = intent.find('tag').text if intent.find('tag') is not None else ''
            patterns = []
            patterns_node = intent.find('patterns')
            if patterns_node is not None:
                for p in patterns_node.findall('pattern'):
                    if p.text:
                        patterns.append(p.text)
            responses = []
            responses_node = intent.find('responses')
            if responses_node is not None:
                for r in responses_node.findall('response'):
                    if r.text:
                        responses.append(r.text)
            data['intents'].append({
                'tag': tag,
                'patterns': patterns,
                'responses': responses
            })
    elif file_path.lower().endswith(('.txt', '.log')):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            data = json.loads(content)
    elif file_path.lower().endswith('.sql'):
        data = {"intents": []}
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        pattern = r"INSERT INTO\s+intents\s*\(.*?\)\s*VALUES\s*\((.*?)\);"
        matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
        for match in matches:
            values = re.findall(r"'(.*?)'", match)
            if len(values) >= 3:
                tag = values[0]
                patterns_str = values[1]
                responses_str = values[2]
                try:
                    patterns = json.loads(patterns_str)
                except Exception:
                    patterns = patterns_str.split(';')
                try:
                    responses = json.loads(responses_str)
                except Exception:
                    responses = responses_str.split(';')
                data['intents'].append({
                    'tag': tag,
                    'patterns': patterns,
                    'responses': responses
                })
    elif file_path.lower().endswith('.db'):
        data = {"intents": []}
        conn = sqlite3.connect(file_path)
        cur = conn.cursor()
        try:
            cur.execute("SELECT * FROM intents")
            rows = cur.fetchall()
            # Asumsikan urutan kolom: tag, patterns, responses (tanpa mempedulikan nama kolom)
            for row in rows:
                if len(row) < 3:
                    continue
                tag, patterns_val, responses_val = row[:3]
                try:
                    patterns = json.loads(patterns_val)
                except Exception:
                    patterns = patterns_val.split(';')
                try:
                    responses = json.loads(responses_val)
                except Exception:
                    responses = responses_val.split(';')
                data['intents'].append({
                    'tag': tag,
                    'patterns': patterns,
                    'responses': responses
                })
        except Exception as e:
            raise ValueError(f"Error reading .db file {file_path}: {e}")
        finally:
            conn.close()
    elif file_path.lower().endswith('.npy'):
        loaded = np.load(file_path, allow_pickle=True)
        if isinstance(loaded, dict) and "intents" in loaded:
            data = loaded
        else:
            data = {"intents": []}
            for item in loaded:
                if isinstance(item, dict) and 'tag' in item and 'patterns' in item and 'responses' in item:
                    data['intents'].append(item)
    else:
        raise ValueError("Format file tidak didukung: " + file_path)
    return data

# Fungsi untuk menggabungkan data dari seluruh file yang ditemukan
def load_all_data(directory):
    data_files = find_data_files(directory)
    if not data_files:
        raise ValueError("Tidak ada file dengan ekstensi yang didukung ditemukan di directory: " + directory)
    combined_data = {"intents": []}
    for file in data_files:
        print(f"\nLoading file: {file}")
        try:
            dataset = load_data(file)
            if "intents" in dataset:
                combined_data["intents"].extend(dataset["intents"])
            else:
                print("File", file, "tidak memiliki key 'intents'")
        except Exception as e:
            print(f"Gagal memuat file: {file} Error: {e}")
    return combined_data

# Fungsi preprocessing: tokenisasi dan pengumpulan kata
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

# Prediksi kelas (tag) untuk input pengguna
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
        responses = [i['responses'] for i in data['intents'] if i['tag'] == tag]
        if responses:
            print("Bot:", random.choice(responses[0]))
        else:
            print("Bot: Maaf, saya tidak mengerti.")

if __name__ == "__main__":
    # Ubah data_directory sesuai kebutuhan; misalnya "./datasets"
    data_directory = "./datasets"  
    data = load_all_data(data_directory)
    words, labels, docs_x, docs_y = preprocess_data(data)
    X_train, y_train = prepare_training_data(words, labels, docs_x, docs_y)
    
    # Pastikan X_train memiliki dimensi yang sesuai sebelum reshape
    if len(X_train.shape) == 2:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    else:
        raise ValueError("X_train tidak memiliki dimensi yang sesuai.")
    
    model = create_model(X_train.shape, len(labels))
    train_model(model, X_train, y_train)
    chatbot(data, words, labels)
