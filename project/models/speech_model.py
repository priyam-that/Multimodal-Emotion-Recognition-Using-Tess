from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Flatten, Bidirectional

def build_speech_model(input_shape, num_classes, return_features=False):
    """
    Build LSTM-based model for speech emotion recognition.
    
    Args:
        input_shape (tuple): Shape of input features (time_steps, n_features).
        num_classes (int): Number of emotion classes.
        return_features (bool): If True, returns the feature vector before classification.
        
    Returns:
        Model: Keras model.
    """
    inputs = Input(shape=input_shape, name='speech_input')
    
    # LSTM layers - Reduced size for i3 11th gen
    x = Bidirectional(LSTM(64, return_sequences=True), name='speech_lstm_1')(inputs)
    x = Dropout(0.3, name='speech_dropout_1')(x)
    x = Bidirectional(LSTM(32), name='speech_lstm_2')(x)
    x = Dropout(0.3, name='speech_dropout_2')(x)
    
    # Dense layers
    x = Dense(32, activation='relu', name='speech_dense_1')(x)
    
    if return_features:
        return Model(inputs=inputs, outputs=x, name='speech_model')
    
    outputs = Dense(num_classes, activation='softmax', name='speech_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='speech_model')
    return model
