import cv2
import numpy as np
from ml_models.gaze_analysis.attention_analyzer import AttentionAnalyzer
from ml_models.body_analysis.pose_estimator import PoseEstimator
from ml_models.facial_analysis.emotion_classifier import EmotionClassifier
from ml_models.authentication.behavioral_authenticator import BehavioralAuthenticator, BehavioralEnroller

class BehavioralAuthSystem:
    def __init__(self):
        # Initialize analysis components
        self.attention_analyzer = AttentionAnalyzer()
        self.pose_estimator = PoseEstimator()
        self.emotion_classifier = EmotionClassifier()
        self.enroller = BehavioralEnroller()
        self.authenticator = BehavioralAuthenticator({})
        self.monitor = ContinuousBehavioralMonitor(self.authenticator, self._security_alert_handler)
        
        # User database (in production, use secure database)
        self.user_templates = {}
    
    def enroll_user(self, user_id: str, enrollment_videos: List[str]):
        """Enroll user from video samples"""
        behavioral_samples = []
        
        for video_path in enrollment_videos:
            # Process enrollment video
            samples = self._process_video_for_enrollment(video_path)
            behavioral_samples.extend(samples)
        
        # Create behavioral template
        template = self.enroller.enroll_user(user_id, behavioral_samples)
        self.user_templates[user_id] = template
        self.authenticator.template_database[user_id] = template
        
        print(f"User {user_id} enrolled successfully")
    
    def authenticate_live(self, user_id: str, camera_index: int = 0):
        """Continuous authentication from webcam"""
        cap = cv2.VideoCapture(camera_index)
        
        print(f"Starting continuous authentication for user: {user_id}")
        self.monitor.start_user_session(user_id)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze current frame
            current_behavior = self._analyze_frame(frame)
            
            # Authenticate
            auth_result = self.monitor.update_behavioral_analysis(user_id, current_behavior, frame)
            
            # Display results
            self._display_authentication_status(frame, auth_result)
            
            # Check if reauthentication is needed
            if self.monitor.require_reauthentication(user_id):
                print("Reauthentication required!")
                # Could trigger additional verification here
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _analyze_frame(self, frame: np.ndarray) -> Dict:
        """Extract behavioral features from frame"""
        behavior_data = {}
        
        # Gaze analysis
        gaze_data = self.attention_analyzer.analyze_gaze_patterns(frame)
        behavior_data['gaze_analysis'] = gaze_data
        
        # Pose analysis
        pose_data = self.pose_estimator.estimate_pose(frame)
        if pose_data:
            behavior_data['body_analysis'] = self.pose_estimator.analyze_posture(pose_data)
        
        # Facial analysis
        emotion_data = self.emotion_classifier.analyze_emotions(frame)
        behavior_data['facial_analysis'] = emotion_data
        
        return behavior_data
    
    def _security_alert_handler(self, user_id: str, alert):
        """Handle security alerts"""
        print(f"SECURITY ALERT - User: {user_id}")
        print(f"Level: {alert.alert_level}, Type: {alert.alert_type}")
        print(f"Description: {alert.description}")
        print(f"Confidence: {alert.confidence:.3f}")
        print("-" * 50)
        
        # In production, you might:
        # - Log to security system
        # - Notify administrators
        # - Require additional verification
        # - Limit user access
    
    def _display_authentication_status(self, frame: np.ndarray, auth_result):
        """Display authentication status on frame"""
        status = "AUTHENTICATED" if auth_result.is_authenticated else "NOT AUTHENTICATED"
        confidence = auth_result.confidence_score
        color = (0, 255, 0) if auth_result.is_authenticated else (0, 0, 255)
        
        cv2.putText(frame, f"Status: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.3f}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Display risk factors
        y_offset = 110
        for risk in auth_result.risk_factors[:3]:  # Show first 3 risk factors
            cv2.putText(frame, f"Risk: {risk}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            y_offset += 30
        
        cv2.imshow('Behavioral Authentication', frame)

# Usage Example
if __name__ == "__main__":
    auth_system = BehavioralAuthSystem()
    
    # Enroll a user (you would collect multiple video samples)
    # auth_system.enroll_user("user123", ["enrollment_video1.mp4", "enrollment_video2.mp4"])
    
    # Continuous authentication
    auth_system.authenticate_live("user123")