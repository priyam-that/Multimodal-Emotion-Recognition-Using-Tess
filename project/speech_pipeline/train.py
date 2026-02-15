import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset_loader import load_tess_data, split_data
from utils.preprocess_speech import extract_mfcc, normalize_features
from models.speech_model import build_speech_model

def train_speech_model(data_path='project/data', epochs=20):
    model_save_path = 'project/models/speech_model.h5'
    encoder_save_path = 'project/models/label_encoder.pkl'
    
    if not os.path.exists('project/models'):
        os.makedirs('project/models')
        
    print("Loading data...")
    df = load_tess_data(data_path)
    if df.empty:
        print("No data found. Please check data path.")
        return

    train_df, val_df, test_df = split_data(df)
    
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    
    # Process audio features
    print("Extracting features...")
    
    # Fixed length for MFCC sequence (e.g. 2s * 100 frames/s = 200)
    # TESS avg length ~2s. Let's use max_len=200 for now.
    MAX_LEN = 200
    
    def process_df(dataframe):
        X = []
        y = []
        for index, row in dataframe.iterrows():
            mfcc = extract_mfcc(row['path'], max_len=MAX_LEN)
            if mfcc is not None:
                X.append(mfcc.T) # Transpose to (timeval, n_mfcc)
                y.append(row['label'])
        return np.array(X), np.array(y)
    
    X_train, y_train = process_df(train_df)
    X_val, y_val = process_df(val_df)
    
    # Normalize features
    # Calculate mean/std from train set and apply to val
    mean = np.mean(X_train, axis=(0, 1))
    std = np.std(X_train, axis=(0, 1))
    
    X_train = (X_train - mean) / (std + 1e-8)
    X_val = (X_val - mean) / (std + 1e-8)
    
    # Save scaler statistics if needed (skipping for now, simple normalization)
    
    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    
    y_train_cat = to_categorical(y_train_enc)
    y_val_cat = to_categorical(y_val_enc)
    
    with open(encoder_save_path, 'wb') as f:
        pickle.dump(le, f)
        
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(le.classes_)
    
    model = build_speech_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    # Train with smaller batch size for i3
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
    plt.savefig('project/Results/plots/speech_training_history.png')
    print("Training history plot saved.")

if __name__ == "__main__":
    train_speech_model()
