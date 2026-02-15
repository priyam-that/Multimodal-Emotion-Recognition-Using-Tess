from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional, GlobalMaxPooling1D

def build_text_model(vocab_size, embedding_dim, max_len, num_classes, embedding_matrix=None, return_features=False):
    """
    Build LSTM-based model for text emotion recognition.
    
    Args:
        vocab_size (int): Size of vocabulary.
        embedding_dim (int): Dimension of word embeddings.
        max_len (int): Maximum length of input sequences.
        num_classes (int): Number of emotion classes.
        embedding_matrix (np.array): Pre-trained embedding matrix (optional).
        return_features (bool): If True, returns the feature vector before classification.
        
    Returns:
        Model: Keras model.
    """
    inputs = Input(shape=(max_len,), name='text_input')
    
    if embedding_matrix is not None:
        embedding_layer = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False, name='text_embedding')(inputs)
    else:
        embedding_layer = Embedding(vocab_size, embedding_dim, name='text_embedding')(inputs)
        
    x = Bidirectional(LSTM(32, return_sequences=True), name='text_lstm')(embedding_layer)
    x = GlobalMaxPooling1D(name='text_global_pool')(x) # Better for text classification often than last state
    x = Dropout(0.3, name='text_dropout')(x)
    x = Dense(16, activation='relu', name='text_dense')(x)
    
    if return_features:
        return Model(inputs=inputs, outputs=x, name='text_model')
        
    outputs = Dense(num_classes, activation='softmax', name='text_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='text_model')
    return model
