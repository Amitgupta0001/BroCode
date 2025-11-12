import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
import warnings

@dataclass
class AnomalyDetection:
    is_anomaly: bool
    anomaly_score: float
    anomaly_type: str
    confidence: float
    features: List[str]

class BehavioralAnomalyDetector:
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = []
        
        # Define normal behavioral ranges (can be learned from data)
        self.normal_ranges = {
            'gaze_stability': (0.3, 0.9),
            'head_movement': (0.1, 0.7),
            'gesture_frequency': (0.5, 5.0),
            'posture_changes': (0.1, 2.0),
            'facial_expressiveness': (0.2, 0.8)
        }
    
    def extract_behavioral_features(self, behavioral_data: Dict) -> np.ndarray:
        """Extract features for anomaly detection"""
        features = []
        feature_names = []
        
        # Gaze and attention features
        if 'attention_metrics' in behavioral_data:
            attention = behavioral_data['attention_metrics']
            features.extend([
                attention.get('attention_score', 0.5),
                attention.get('gaze_stability', 0.5),
                attention.get('distraction_frequency', 0.0)
            ])
            feature_names.extend(['attention_score', 'gaze_stability', 'distraction_freq'])
        
        # Movement and posture features
        if 'movement_metrics' in behavioral_data:
            movement = behavioral_data['movement_metrics']
            features.extend([
                movement.get('activity_level', 0.0),
                movement.get('movement_smoothness', 1.0),
                movement.get('fidgeting_score', 0.0)
            ])
            feature_names.extend(['activity_level', 'movement_smoothness', 'fidgeting_score'])
        
        # Facial expression features
        if 'facial_metrics' in behavioral_data:
            facial = behavioral_data['facial_metrics']
            features.extend([
                facial.get('emotion_variability', 0.0),
                facial.get('micro_expression_frequency', 0.0),
                facial.get('expressiveness', 0.5)
            ])
            feature_names.extend(['emotion_variability', 'micro_expr_freq', 'expressiveness'])
        
        # Temporal consistency features
        if 'temporal_metrics' in behavioral_data:
            temporal = behavioral_data['temporal_metrics']
            features.extend([
                temporal.get('behavior_consistency', 0.8),
                temporal.get('pattern_regularity', 0.5)
            ])
            feature_names.extend(['behavior_consistency', 'pattern_regularity'])
        
        self.feature_names = feature_names
        return np.array(features).reshape(1, -1)
    
    def detect_anomaly(self, behavioral_data: Dict) -> AnomalyDetection:
        """Detect behavioral anomalies"""
        if not self.is_fitted:
            return self._rule_based_detection(behavioral_data)
        
        try:
            # Extract features
            features = self.extract_behavioral_features(behavioral_data)
            features_scaled = self.scaler.transform(features)
            
            # Predict anomaly
            is_anomaly = self.model.predict(features_scaled)[0] == -1
            anomaly_score = -self.model.score_samples(features_scaled)[0]
            
            # Determine anomaly type and confidence
            anomaly_type = self._classify_anomaly_type(features[0])
            confidence = self._calculate_confidence(anomaly_score, features[0])
            
            return AnomalyDetection(
                is_anomaly=bool(is_anomaly),
                anomaly_score=float(anomaly_score),
                anomaly_type=anomaly_type,
                confidence=float(confidence),
                features=self.feature_names
            )
            
        except Exception as e:
            warnings.warn(f"Anomaly detection failed: {e}")
            return self._rule_based_detection(behavioral_data)
    
    def _rule_based_detection(self, behavioral_data: Dict) -> AnomalyDetection:
        """Fallback rule-based anomaly detection"""
        anomaly_score = 0.0
        anomaly_flags = []
        
        # Check individual metrics against normal ranges
        if 'attention_metrics' in behavioral_data:
            attention = behavioral_data['attention_metrics']
            if attention.get('attention_score', 0.5) < 0.2:
                anomaly_score += 0.3
                anomaly_flags.append('low_attention')
            if attention.get('distraction_frequency', 0) > 10:
                anomaly_score += 0.2
                anomaly_flags.append('high_distraction')
        
        if 'movement_metrics' in behavioral_data:
            movement = behavioral_data['movement_metrics']
            if movement.get('fidgeting_score', 0) > 0.5:
                anomaly_score += 0.3
                anomaly_flags.append('high_fidgeting')
            if movement.get('activity_level', 0) > 0.9:
                anomaly_score += 0.2
                anomaly_flags.append('hyperactive')
        
        is_anomaly = anomaly_score > 0.5
        anomaly_type = 'multiple' if len(anomaly_flags) > 1 else anomaly_flags[0] if anomaly_flags else 'normal'
        
        return AnomalyDetection(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            anomaly_type=anomaly_type,
            confidence=min(anomaly_score, 1.0),
            features=['rule_based']
        )
    
    def _classify_anomaly_type(self, features: np.ndarray) -> str:
        """Classify the type of behavioral anomaly"""
        if len(features) != len(self.feature_names):
            return 'unknown'
        
        feature_dict = dict(zip(self.feature_names, features))
        
        # Define anomaly type thresholds
        if feature_dict.get('fidgeting_score', 0) > 0.7:
            return 'nervous_energy'
        elif feature_dict.get('attention_score', 0.5) < 0.2:
            return 'inattentive'
        elif feature_dict.get('activity_level', 0) > 0.8:
            return 'hyperactive'
        elif feature_dict.get('gaze_stability', 0.5) < 0.2:
            return 'distracted'
        elif feature_dict.get('emotion_variability', 0) > 0.8:
            return 'emotionally_unstable'
        else:
            return 'behavioral_irregularity'
    
    def _calculate_confidence(self, anomaly_score: float, features: np.ndarray) -> float:
        """Calculate confidence in anomaly detection"""
        base_confidence = anomaly_score
        
        # Increase confidence for extreme values
        extreme_features = 0
        for feature, value in zip(self.feature_names, features):
            if feature in self.normal_ranges:
                low, high = self.normal_ranges[feature]
                if value < low * 0.5 or value > high * 1.5:
                    extreme_features += 1
        
        confidence_boost = extreme_features * 0.1
        return min(base_confidence + confidence_boost, 1.0)
    
    def fit(self, behavioral_data_list: List[Dict]):
        """Fit the anomaly detection model on normal behavioral data"""
        if not behavioral_data_list:
            raise ValueError("No behavioral data provided for training")
        
        # Extract features from all samples
        features_list = []
        for data in behavioral_data_list:
            features = self.extract_behavioral_features(data)
            features_list.append(features.flatten())
        
        X = np.array(features_list)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        print(f"Anomaly detector fitted on {len(X)} samples with {X.shape[1]} features")
    
    def update_normal_ranges(self, behavioral_data_list: List[Dict]):
        """Update normal behavioral ranges from data"""
        if not behavioral_data_list:
            return
        
        # Collect all feature values
        feature_accumulator = {name: [] for name in self.normal_ranges.keys()}
        
        for data in behavioral_data_list:
            if 'attention_metrics' in data:
                attention = data['attention_metrics']
                if 'gaze_stability' in attention:
                    feature_accumulator['gaze_stability'].append(attention['gaze_stability'])
            
            if 'movement_metrics' in data:
                movement = data['movement_metrics']
                if 'activity_level' in movement:
                    feature_accumulator['head_movement'].append(movement['activity_level'])
                if 'gesture_frequency' in movement:
                    feature_accumulator['gesture_frequency'].append(movement['gesture_frequency'])
        
        # Update ranges (mean ± 2 standard deviations)
        for feature_name, values in feature_accumulator.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                self.normal_ranges[feature_name] = (
                    max(0, mean_val - 2 * std_val),
                    min(1, mean_val + 2 * std_val)
                )