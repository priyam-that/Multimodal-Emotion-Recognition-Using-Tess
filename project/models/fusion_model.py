from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout

def build_fusion_model(speech_model, text_model, num_classes):
    """
    Build a fusion model by combining speech and text models.
    
    Args:
        speech_model (Model): Pre-trained speech model (or part of it).
        text_model (Model): Pre-trained text model (or part of it).
        num_classes (int): Number of emotion classes.
        
    Returns:
        Model: Keras model.
    """
    # Use the outputs of the sub-models as inputs to the fusion layers
    # Assumption: speech_model and text_model already return feature vectors (not probabilities)
    # If they return probabilities, we might want to pop the last layer or use a new model that outputs features.
    
    speech_features = speech_model.output
    text_features = text_model.output
    
    # Concatenate features
    combined = Concatenate()([speech_features, text_features])
    
    # Fusion layers - Reduced size
    x = Dense(32, activation='relu')(combined)
    x = Dropout(0.3)(x)
    x = Dense(16, activation='relu')(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=[speech_model.input, text_model.input], outputs=outputs)
    return model
