import cv2
import numpy as np
from typing import List, Dict
from ML.ml_models.gaze_analysis.attention_analyzer import AttentionAnalyzer
from ML.ml_models.body_analysis.pose_estimator import PoseEstimator
from ML.ml_models.facial_analysis.emotion_classifier import EmotionClassifier
from ML.ml_models.authentication.behavioral_authenticator import BehavioralAuthenticator, BehavioralEnroller
from ML.ml_models.authentication.continuous_monitor import ContinuousBehavioralMonitor
from ML.fusion_engine import FusionEngine

class BehavioralAuthSystem:
    """
    Core orchestrator for BroCode Adaptive Behavioral Authentication.
    Combines gaze, pose, facial, and keystroke signals into a unified trust score.
    """

    def __init__(self):
        # Initialize core modules
        self.attention_analyzer = AttentionAnalyzer()
        self.pose_estimator = PoseEstimator()
        self.emotion_classifier = EmotionClassifier()
        self.enroller = BehavioralEnroller()
        self.authenticator = BehavioralAuthenticator({})
        self.fusion_engine = FusionEngine()
        self.monitor = ContinuousBehavioralMonitor(self._security_alert_handler)

        # Local user templates (can be replaced by DB)
        self.user_templates = {}

    # -------------------- Enrollment --------------------
    def enroll_user(self, user_id: str, enrollment_videos: List[str]):
        """Enroll a user using video samples."""
        behavioral_samples = []
        for video_path in enrollment_videos:
            samples = self._process_video_for_enrollment(video_path)
            behavioral_samples.extend(samples)
        template = self.enroller.enroll_user(user_id, behavioral_samples)
        self.user_templates[user_id] = template
        self.authenticator.template_database[user_id] = template
        print(f"[INFO] User {user_id} enrolled successfully.")

    # -------------------- Continuous Auth --------------------
    def authenticate_live(self, user_id: str, camera_index: int = 0):
        """Real-time continuous authentication with adaptive monitoring."""
        cap = cv2.VideoCapture(camera_index)
        print(f"[INFO] Starting continuous authentication for user: {user_id}")
        self.monitor.start_session(user_id)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                behavior = self._analyze_frame(frame)
                trust_score, anomaly = self.monitor.update_behavioral_analysis(user_id, behavior)
            except Exception as e:
                print(f"[WARN] Frame analysis failed: {e}")
                trust_score, anomaly = 0.5, False

            # Display
            self._display_authentication_status(frame, trust_score, anomaly)

            if self.monitor.require_reauth(user_id):
                print("⚠️  Reauthentication required due to trust drop!")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # -------------------- Frame Analysis --------------------
    def _analyze_frame(self, frame: np.ndarray) -> Dict:
        """Extract and normalize behavioral metrics from frame."""
        behavior_data = {}

        # Gaze analysis
        gaze_data = self.attention_analyzer.analyze_gaze_patterns(frame)
        behavior_data["attention_score"] = gaze_data.get("attention_score", 0.6)
        behavior_data["gaze_stability"] = gaze_data.get("stability", 0.6)

        # Pose analysis
        pose_data = self.pose_estimator.estimate_pose(frame)
        if pose_data:
            pose_eval = self.pose_estimator.analyze_posture(pose_data)
            behavior_data["movement_smoothness"] = pose_eval.get("smoothness", 0.7)

        # Facial emotion analysis
        emotion_data = self.emotion_classifier.analyze_emotions(frame)
        behavior_data["emotion_variability"] = emotion_data.get("variability", 0.5)

        # Combined trust estimate (optional early fusion)
        behavior_data["trust_score"] = self.fusion_engine.compute_trust_score(behavior_data)
        return behavior_data

    # -------------------- Flask Integration --------------------
    def monitor_session(self, user_id: str, frame_data: Dict, keystrokes: List[Dict]):
        """
        Interface for Flask /monitor_activity endpoint.
        Combines multi-modal data and returns (trust_score, anomaly_flag).
        """
        try:
            # Merge web-sent frame_data and simulated keystroke trust
            behavior = {
                "attention_score": frame_data.get("gaze_score", 0.6),
                "gaze_stability": frame_data.get("pose_score", 0.7),
                "movement_smoothness": frame_data.get("pose_score", 0.8),
                "emotion_variability": frame_data.get("emotion_score", 0.5),
                "trust_score": frame_data.get("frame_trust", 0.7),
            }
            trust, anomaly = self.monitor.update_behavioral_analysis(user_id, behavior)
            return trust, anomaly
        except Exception as e:
            print(f"[ERROR] monitor_session failed: {e}")
            return 0.5, False

    # -------------------- Security Alerts --------------------
    def _security_alert_handler(self, user_id: str, alert):
        print(f"\n🚨 SECURITY ALERT – User: {user_id}")
        print(f"Level: {alert.alert_level}, Type: {alert.alert_type}")
        print(f"Description: {alert.description}")
        print(f"Confidence: {alert.confidence:.3f}")
        print("-" * 50)

    # -------------------- Display --------------------
    def _display_authentication_status(self, frame, trust_score, anomaly_flag):
        """Render status overlay on webcam feed."""
        color = (0, 255, 0) if trust_score >= 0.6 else (0, 0, 255)
        status = "Authenticated" if trust_score >= 0.6 else "At Risk"

        cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Trust: {trust_score:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if anomaly_flag:
            cv2.putText(frame, "⚠️ Behavioral Anomaly Detected", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("BroCode Behavioral Auth", frame)
