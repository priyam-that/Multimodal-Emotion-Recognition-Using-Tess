import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset_loader import load_tess_data, split_data
from utils.preprocess_speech import extract_mfcc, normalize_features

def test_speech_model():
    data_path = 'project/data'
    model_path = 'project/models/speech_model.h5'
    encoder_path = 'project/models/label_encoder.pkl'
    
    if not os.path.exists(model_path):
        print("Model not found. Please train first.")
        return
        
    print("Loading data...")
    df = load_tess_data(data_path)
    _, _, test_df = split_data(df)
    
    # Process audio features
    print("Extracting features...")
    MAX_LEN = 200
    
    X_test = []
    y_test = []
    for index, row in test_df.iterrows():
        mfcc = extract_mfcc(row['path'], max_len=MAX_LEN)
        if mfcc is not None:
            X_test.append(mfcc.T)
            y_test.append(row['label'])
            
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Normalize (using simple standardization based on this batch, or ideally saved scaler)
    # For now, per-sample normalization or just simple standardization
    mean = np.mean(X_test, axis=(0, 1))
    std = np.std(X_test, axis=(0, 1))
    X_test = (X_test - mean) / (std + 1e-8)
    
    # Load Encoder
    with open(encoder_path, 'rb') as f:
        le = pickle.load(f)
    y_test_enc = le.transform(y_test)
    
    # Load Model
    model = tf.keras.models.load_model(model_path)
    
    # Predict
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Metrics
    print(classification_report(y_test_enc, y_pred, target_names=le.classes_))
    
    cm = confusion_matrix(y_test_enc, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Speech Model Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if not os.path.exists('project/Results/plots'):
        os.makedirs('project/Results/plots')
    plt.savefig('project/Results/plots/speech_confusion_matrix.png')
    print("Confusion matrix saved.")

if __name__ == "__main__":
    test_speech_model()
