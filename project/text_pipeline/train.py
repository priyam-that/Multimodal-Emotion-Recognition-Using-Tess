import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset_loader import load_tess_data, split_data
from utils.preprocess_text import clean_text, create_tokenizer, preprocess_texts
from models.text_model import build_text_model

def train_text_model(data_path='project/data', epochs=15):
    model_save_path = 'project/models/text_model.h5'
    tokenizer_save_path = 'project/models/tokenizer.pkl'
    encoder_save_path = 'project/models/label_encoder.pkl'
    
    if not os.path.exists('project/models'):
        os.makedirs('project/models')
        
    print("Loading data...")
    df = load_tess_data(data_path)
    if df.empty:
        print("No data found.")
        return

    train_df, val_df, test_df = split_data(df)
    
    # Preprocess text
    print("Preprocessing text...")
    
    # Clean text
    train_texts = [clean_text(t) for t in train_df['transcript']]
    val_texts = [clean_text(t) for t in val_df['transcript']]
    
    # Fit tokenizer on train texts
    VOCAB_SIZE = 1000 # TESS has small vocab
    tokenizer = create_tokenizer(train_texts, num_words=VOCAB_SIZE)
    
    with open(tokenizer_save_path, 'wb') as f:
        pickle.dump(tokenizer, f)
        
    # Convert to sequences
    MAX_LEN = 20 # TESS sentences are short "Say the word X"
    X_train = preprocess_texts(train_texts, tokenizer, max_len=MAX_LEN)
    X_val = preprocess_texts(val_texts, tokenizer, max_len=MAX_LEN)
    
    # Encode labels
    # Load existing encoder if available (from speech training) to ensure consistency
    if os.path.exists(encoder_save_path):
        with open(encoder_save_path, 'rb') as f:
            le = pickle.load(f)
        y_train_enc = le.transform(train_df['label'])
        y_val_enc = le.transform(val_df['label'])
    else:
        le = LabelEncoder()
        y_train_enc = le.fit_transform(train_df['label'])
        y_val_enc = le.transform(val_df['label'])
        with open(encoder_save_path, 'wb') as f:
            pickle.dump(le, f)
            
    y_train_cat = to_categorical(y_train_enc)
    y_val_cat = to_categorical(y_val_enc)
    
    # Build model
    EMBEDDING_DIM = 50
    num_classes = len(unique_labels := np.unique(y_train_enc)) if 'le' not in locals() else len(le.classes_)
    vocab_size = min(len(tokenizer.word_index) + 1, VOCAB_SIZE)
    
    model = build_text_model(vocab_size, EMBEDDING_DIM, MAX_LEN, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    # Train with smaller batch size
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=16
    )
    
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Plot history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Loss')
    plt.legend()
    
    if not os.path.exists('project/Results/plots'):
        os.makedirs('project/Results/plots')
    plt.savefig('project/Results/plots/text_training_history.png')

if __name__ == "__main__":
    train_text_model()
