import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
from typing import Dict, List

class EmotionClassifier:
    def __init__(self):
        self.model = self._build_emotion_model()
        self.emotion_labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger']
        self.emotion_history = []
        self.max_history = 30
    
    def _build_emotion_model(self) -> models.Sequential:
        """Build CNN model for emotion classification"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(7, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def preprocess_face(self, face_roi: np.ndarray) -> np.ndarray:
        """Preprocess face ROI for emotion classification"""
        # Convert to grayscale
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize to model input size
        resized = cv2.resize(gray, (48, 48))
        
        # Normalize pixel values
        normalized = resized.astype('float32') / 255.0
        
        # Add channel and batch dimensions
        processed = np.expand_dims(normalized, axis=-1)
        processed = np.expand_dims(processed, axis=0)
        
        return processed
    
    def predict_emotion(self, face_roi: np.ndarray) -> Dict:
        """Predict emotion from face ROI"""
        try:
            processed_face = self.preprocess_face(face_roi)
            predictions = self.model.predict(processed_face, verbose=0)
            
            emotion_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][emotion_idx])
            emotion = self.emotion_labels[emotion_idx]
            
            # Update emotion history
            self._update_emotion_history(emotion, confidence)
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'all_predictions': dict(zip(self.emotion_labels, predictions[0].tolist()))
            }
        except Exception as e:
            print(f"Error in emotion prediction: {e}")
            return {'emotion': 'unknown', 'confidence': 0.0}
    
    def get_emotion_consistency(self) -> float:
        """Calculate consistency of emotions over time"""
        if len(self.emotion_history) < 2:
            return 1.0
        
        emotions = [entry['emotion'] for entry in self.emotion_history]
        unique_emotions = len(set(emotions))
        consistency = 1.0 - (unique_emotions / len(emotions))
        
        return max(0, min(1, consistency))
    
    def _update_emotion_history(self, emotion: str, confidence: float):
        """Update emotion history for temporal analysis"""
        self.emotion_history.append({
            'emotion': emotion,
            'confidence': confidence,
            'timestamp': np.datetime64('now')
        })
        
        # Keep only recent history
        if len(self.emotion_history) > self.max_history:
            self.emotion_history.pop(0)