import numpy as np
from typing import Dict, List, Optional, Callable
from collections import deque
import time
from dataclasses import dataclass

@dataclass
class SecurityAlert:
    alert_level: str  # 'low', 'medium', 'high'
    alert_type: str
    description: str
    confidence: float
    timestamp: float

class ContinuousBehavioralMonitor:
    def __init__(self, authenticator: BehavioralAuthenticator, alert_callback: Optional[Callable] = None):
        self.authenticator = authenticator
        self.alert_callback = alert_callback
        self.user_sessions = {}
        self.alert_history = deque(maxlen=100)
        
    def start_user_session(self, user_id: str, initial_confidence: float = 0.8):
        """Start monitoring a user session"""
        self.user_sessions[user_id] = {
            'start_time': time.time(),
            'last_authentication': time.time(),
            'continuous_confidence': initial_confidence,
            'authentication_history': [],
            'risk_level': 'low'
        }
    
    def update_behavioral_analysis(self, user_id: str, current_behavior: Dict, video_frame: np.ndarray = None):
        """Update behavioral analysis and check for anomalies"""
        if user_id not in self.user_sessions:
            self.start_user_session(user_id)
        
        # Perform authentication check
        auth_result = self.authenticator.authenticate_user(user_id, current_behavior, video_frame)
        
        # Update session data
        session = self.user_sessions[user_id]
        session['last_authentication'] = time.time()
        session['authentication_history'].append(auth_result)
        
        # Update continuous confidence (exponential moving average)
        alpha = 0.3  # Smoothing factor
        session['continuous_confidence'] = (
            alpha * auth_result.confidence_score + 
            (1 - alpha) * session['continuous_confidence']
        )
        
        # Check for security alerts
        self._check_security_alerts(user_id, auth_result, session)
        
        return auth_result
    
    def _check_security_alerts(self, user_id: str, auth_result: AuthenticationResult, session: Dict):
        """Check for various security alert conditions"""
        alerts = []
        
        # 1. Low confidence alert
        if auth_result.confidence_score < 0.3:
            alerts.append(SecurityAlert(
                alert_level='high',
                alert_type='low_confidence',
                description=f"Very low behavioral confidence: {auth_result.confidence_score:.3f}",
                confidence=1.0 - auth_result.confidence_score,
                timestamp=time.time()
            ))
        
        # 2. Sudden behavior change
        if len(session['authentication_history']) >= 3:
            recent_scores = [auth.confidence_score for auth in session['authentication_history'][-3:]]
            if np.std(recent_scores) > 0.3:  # High variance in recent scores
                alerts.append(SecurityAlert(
                    alert_level='medium',
                    alert_type='behavior_instability',
                    description="Sudden changes in behavioral patterns detected",
                    confidence=0.7,
                    timestamp=time.time()
                ))
        
        # 3. Multiple risk factors
        if len(auth_result.risk_factors) >= 2:
            alerts.append(SecurityAlert(
                alert_level='medium',
                alert_type='multiple_risk_factors',
                description=f"Multiple behavioral risk factors: {auth_result.risk_factors}",
                confidence=0.8,
                timestamp=time.time()
            ))
        
        # 4. Continuous degradation
        if len(session['authentication_history']) >= 5:
            recent_scores = [auth.confidence_score for auth in session['authentication_history'][-5:]]
            if np.polyfit(range(5), recent_scores, 1)[0] < -0.1:  # Negative trend
                alerts.append(SecurityAlert(
                    alert_level='low',
                    alert_type='confidence_degradation',
                    description="Gradual decline in behavioral confidence detected",
                    confidence=0.6,
                    timestamp=time.time()
                ))
        
        # Trigger alert callbacks
        for alert in alerts:
            self.alert_history.append(alert)
            if self.alert_callback:
                self.alert_callback(user_id, alert)
    
    def get_session_risk_level(self, user_id: str) -> str:
        """Get current risk level for user session"""
        if user_id not in self.user_sessions:
            return 'unknown'
        
        session = self.user_sessions[user_id]
        confidence = session['continuous_confidence']
        
        if confidence > 0.7:
            return 'low'
        elif confidence > 0.5:
            return 'medium'
        else:
            return 'high'
    
    def require_reauthentication(self, user_id: str) -> bool:
        """Determine if reauthentication is required"""
        if user_id not in self.user_sessions:
            return True
        
        session = self.user_sessions[user_id]
        time_since_auth = time.time() - session['last_authentication']
        risk_level = self.get_session_risk_level(user_id)
        
        # Reauthentication intervals based on risk level
        intervals = {'low': 300, 'medium': 120, 'high': 30}  # seconds
        
        return time_since_auth > intervals.get(risk_level, 120)