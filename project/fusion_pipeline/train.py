import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model, Model
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset_loader import load_tess_data, split_data
from utils.preprocess_speech import extract_mfcc, normalize_features
from utils.preprocess_text import clean_text, preprocess_texts, create_tokenizer
from models.fusion_model import build_fusion_model

def train_fusion_model(data_path='project/data', epochs=15):
    speech_model_path = 'project/models/speech_model.h5'
    text_model_path = 'project/models/text_model.h5'
    fusion_model_save_path = 'project/models/fusion_model.h5'
    tokenizer_path = 'project/models/tokenizer.pkl'
    encoder_path = 'project/models/label_encoder.pkl'
    
    if not os.path.exists(speech_model_path) or not os.path.exists(text_model_path):
        print("Please train speech and text models first.")
        return
        
    print("Loading data...")
    df = load_tess_data(data_path)
    train_df, val_df, test_df = split_data(df)
    
    # Process Speech Data
    print("Processing speech data...")
    MAX_LEN_SPEECH = 200
    
    def process_speech(dataframe):
        X = []
        for index, row in dataframe.iterrows():
            mfcc = extract_mfcc(row['path'], max_len=MAX_LEN_SPEECH)
            if mfcc is not None:
                X.append(mfcc.T)
            else:
                # Handle error, maybe append zeros or skip row
                # Ideally skip, but for now append zeros to match size
                X.append(np.zeros((MAX_LEN_SPEECH, 40)))
        return np.array(X)

    X_train_speech = process_speech(train_df)
    X_val_speech = process_speech(val_df)
    
    # Normalize speech features (using same logic as train.py)
    mean = np.mean(X_train_speech, axis=(0, 1))
    std = np.std(X_train_speech, axis=(0, 1))
    X_train_speech = (X_train_speech - mean) / (std + 1e-8)
    X_val_speech = (X_val_speech - mean) / (std + 1e-8)
    
    # Process Text Data
    print("Processing text data...")
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
        
    train_texts = [clean_text(t) for t in train_df['transcript']]
    val_texts = [clean_text(t) for t in val_df['transcript']]
    
    MAX_LEN_TEXT = 20
    X_train_text = preprocess_texts(train_texts, tokenizer, max_len=MAX_LEN_TEXT)
    X_val_text = preprocess_texts(val_texts, tokenizer, max_len=MAX_LEN_TEXT)
    
    # Labels
    with open(encoder_path, 'rb') as f:
        le = pickle.load(f)
        
    y_train_cat = to_categorical(le.transform(train_df['label']))
    y_val_cat = to_categorical(le.transform(val_df['label']))
    
    # Load Models
    print("Loading pre-trained models...")
    speech_model_full = load_model(speech_model_path)
    text_model_full = load_model(text_model_path)
    
    # Remove top classification layers to get features
    # Speech model structure: Input -> [Layers] -> Dense(64) -> Dense(Softmax)
    # We want output of Dense(64). It is layer index -2.
    speech_feature_model = Model(inputs=speech_model_full.input, outputs=speech_model_full.layers[-2].output)
    text_feature_model = Model(inputs=text_model_full.input, outputs=text_model_full.layers[-2].output)
    
    # Freeze base models
    for layer in speech_feature_model.layers:
        layer.trainable = False
    for layer in text_feature_model.layers:
        layer.trainable = False
        
    # Build Fusion Model
    num_classes = len(le.classes_)
    model = build_fusion_model(speech_feature_model, text_feature_model, num_classes)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    # Train with smaller batch size
    history = model.fit(
        [X_train_speech, X_train_text], y_train_cat,
        validation_data=([X_val_speech, X_val_text], y_val_cat),
        epochs=epochs,
        batch_size=16
    )
    
    model.save(fusion_model_save_path)
    print(f"Fusion model saved to {fusion_model_save_path}")
    
    # Plot history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title('Fusion Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Fusion Loss')
    plt.legend()
    
    plt.savefig('project/Results/plots/fusion_training_history.png')

if __name__ == "__main__":
    train_fusion_model()
