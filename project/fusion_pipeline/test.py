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
from utils.preprocess_text import clean_text, preprocess_texts

def test_fusion_model():
    data_path = 'project/data'
    model_path = 'project/models/fusion_model.h5'
    tokenizer_path = 'project/models/tokenizer.pkl'
    encoder_path = 'project/models/label_encoder.pkl'
    
    if not os.path.exists(model_path):
        print("Model not found. Please train first.")
        return
        
    print("Loading data...")
    df = load_tess_data(data_path)
    _, _, test_df = split_data(df)
    
    # Process Speech Data
    print("Processing speech data...")
    MAX_LEN_SPEECH = 200
    
    X_test_speech = []
    y_test = []
    
    for index, row in test_df.iterrows():
        mfcc = extract_mfcc(row['path'], max_len=MAX_LEN_SPEECH)
        if mfcc is not None:
            X_test_speech.append(mfcc.T)
            y_test.append(row['label'])
            
    X_test_speech = np.array(X_test_speech)
    mean = np.mean(X_test_speech, axis=(0, 1))
    std = np.std(X_test_speech, axis=(0, 1))
    X_test_speech = (X_test_speech - mean) / (std + 1e-8)
    
    # Process Text Data
    print("Processing text data...")
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
        
    test_texts = [clean_text(t) for t in test_df['transcript']]
    
    MAX_LEN_TEXT = 20
    X_test_text = preprocess_texts(test_texts, tokenizer, max_len=MAX_LEN_TEXT)
    
    # Load Encoder
    with open(encoder_path, 'rb') as f:
        le = pickle.load(f)
    print(list(le.classes_))
    y_test_enc = le.transform(y_test)
    
    # Load Model
    # Since we used custom model creation in fusion, loading might require custom_objects if we had custom layers.
    # But here we just combined standard layers.
    # However, saving a model with inputs from other models might save the whole graph.
    model = tf.keras.models.load_model(model_path)
    
    # Predict
    # Model expects [speech_input, text_input]
    y_pred_probs = model.predict([X_test_speech, X_test_text])
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Metrics
    print(classification_report(y_test_enc, y_pred, target_names=le.classes_))
    
    cm = confusion_matrix(y_test_enc, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Fusion Model Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if not os.path.exists('project/Results/plots'):
        os.makedirs('project/Results/plots')
    plt.savefig('project/Results/plots/fusion_confusion_matrix.png')
    print("Confusion matrix saved.")

if __name__ == "__main__":
    test_fusion_model()
