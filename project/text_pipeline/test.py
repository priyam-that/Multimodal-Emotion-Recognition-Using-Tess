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
from utils.preprocess_text import clean_text, preprocess_texts

def test_text_model():
    data_path = 'project/data'
    model_path = 'project/models/text_model.h5'
    tokenizer_path = 'project/models/tokenizer.pkl'
    encoder_path = 'project/models/label_encoder.pkl'
    
    if not os.path.exists(model_path):
        print("Model not found. Please train first.")
        return
        
    print("Loading data...")
    df = load_tess_data(data_path)
    _, _, test_df = split_data(df)
    
    # Process text
    print("Preprocessing text...")
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
        
    test_texts = [clean_text(t) for t in test_df['transcript']]
    
    MAX_LEN = 20
    X_test = preprocess_texts(test_texts, tokenizer, max_len=MAX_LEN)
    
    # Load Encoder
    with open(encoder_path, 'rb') as f:
        le = pickle.load(f)
    print(list(le.classes_))
    y_test_enc = le.transform(test_df['label'])
    
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
    plt.title('Text Model Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if not os.path.exists('project/Results/plots'):
        os.makedirs('project/Results/plots')
    plt.savefig('project/Results/plots/text_confusion_matrix.png')
    print("Confusion matrix saved.")

if __name__ == "__main__":
    test_text_model()
