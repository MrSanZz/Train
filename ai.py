"""
AI System dengan PyTorch dan TensorFlow (Fixed Version)
Mencakup:
1. CNN untuk klasifikasi gambar menggunakan PyTorch
2. LSTM untuk generasi teks menggunakan TensorFlow
"""

import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# PyTorch Section
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# TensorFlow Section
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer

# =====================
# PyTorch CNN untuk MNIST (Diperbarui)
# =====================

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ImprovedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train_pytorch_model(device, model, dataloader, criterion, optimizer, epochs=10):
    train_losses = []
    val_losses = []
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader['train']:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(dataloader['train'].dataset)
        train_acc = correct / total
        train_losses.append(train_loss)
        
        val_loss, val_acc = evaluate_pytorch_model(device, model, dataloader['val'], criterion)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print('Model terbaik disimpan')
    
    plot_training_curves(train_losses, val_losses)
    return model

def evaluate_pytorch_model(device, model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(dataloader.dataset)
    val_acc = correct / total
    return val_loss, val_acc

# =====================
# TensorFlow LSTM untuk Generasi Teks (Diperbarui)
# =====================

class TextGenerator(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim=256, lstm_units=512):
        super(TextGenerator, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm1 = LSTM(lstm_units, return_sequences=True)
        self.lstm2 = LSTM(lstm_units)
        self.dense1 = Dense(lstm_units, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.4)
        self.dense2 = Dense(vocab_size, activation='softmax')

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.lstm1(x)
        x = self.lstm2(x)
        if training:
            x = self.dropout(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def prepare_tensorflow_data(corpus, seq_length=100):
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts([corpus])
    
    sequences = tokenizer.texts_to_sequences([corpus])[0]
    vocab_size = len(tokenizer.word_index) + 1
    
    input_sequences = []
    target_chars = []
    
    for i in range(len(sequences) - seq_length):
        seq = sequences[i:i+seq_length]
        target = sequences[i+seq_length]
        input_sequences.append(seq)
        target_chars.append(target)
    
    X = np.array(input_sequences)
    y = np.array(target_chars)
    
    return X, y, tokenizer, vocab_size

def generate_text_improved(model, tokenizer, seed_text, num_gen_chars=500, temperature=0.7):
    generated = []
    input_seq = tokenizer.texts_to_sequences([seed_text])[0]
    
    for _ in range(num_gen_chars):
        encoded = tf.keras.preprocessing.sequence.pad_sequences(
            [input_seq], maxlen=100, truncating='pre')
        
        preds = model.predict(encoded, verbose=0)[0]
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        
        next_index = np.random.choice(len(preds), p=preds)
        input_seq.append(next_index)
        input_seq = input_seq[1:]
        
        generated.append(next_index)
    
    return seed_text + tokenizer.sequences_to_texts([generated])[0]

# =====================
# Utilities (Diperbarui)
# =====================

def plot_training_curves(train_loss, val_loss):
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_curves.png')
    plt.close()

def prepare_pytorch_data(batch_size=128):
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1,0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        transform=transforms.ToTensor()
    )
    
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    dataloaders = {
        'train': DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(val_subset, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    }
    return dataloaders

# =====================
# Main Execution (Diperbarui)
# =====================

def main(args):
    if args.framework == 'pytorch':
        # Inisialisasi PyTorch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Menggunakan perangkat: {device}")
        
        # Persiapan data
        dataloaders = prepare_pytorch_data(args.batch_size)
        
        # Inisialisasi model
        model = ImprovedCNN().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
        criterion = nn.CrossEntropyLoss()
        
        # Pelatihan
        model = train_pytorch_model(
            device, model, dataloaders, criterion, optimizer, args.epochs
        )
        
        # Evaluasi akhir
        test_loss, test_acc = evaluate_pytorch_model(device, model, dataloaders['test'], criterion)
        print(f'\nHasil Tes Akhir: Loss={test_loss:.4f}, Akurasi={test_acc:.4f}')
    
    elif args.framework == 'tensorflow':
        # Inisialisasi TensorFlow
        print("Mempersiapkan data untuk LSTM...")
        corpus = open('shakespeare.txt').read().lower()
        
        X, y, tokenizer, vocab_size = prepare_tensorflow_data(corpus)
        
        model = TextGenerator(vocab_size)
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3),
            tf.keras.callbacks.ModelCheckpoint('text_gen_best.h5', save_best_only=True)
        ]
        
        # Pelatihan
        history = model.fit(
            X, y,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_split=0.2,
            callbacks=callbacks
        )
        
        # Generasi teks
        seed = "shall i compare thee to a summer's day?\n"
        generated_text = generate_text_improved(model, tokenizer, seed)
        print("\nHasil Generasi Teks:")
        print(generated_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sistem AI Multi-Framework')
    parser.add_argument('--framework', required=True, choices=['pytorch', 'tensorflow'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    
    args = parser.parse_args()
    
    start_time = time.time()
    main(args)
    print(f"\nTotal waktu eksekusi: {time.time() - start_time:.2f} detik")
