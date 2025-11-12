// ML/static/monitor.js
// Captures keystroke events, bundles them and sends to /monitor_activity periodically.
// Also updates the dashboard UI (elements with IDs 'trust' and 'risk').

let gazeScore = 0.5;
let poseScore = 0.5;

(function () {
  // Buffer of keystroke events
  let keystrokes = [];

  // Capture keystrokes globally
  document.addEventListener("keydown", function (e) {
    keystrokes.push({ key: e.key, t: Date.now(), type: "keydown" });
  });

  document.addEventListener("keyup", function (e) {
    keystrokes.push({ key: e.key, t: Date.now(), type: "keyup" });
  });

  // Optionally send small device info (userAgent, language)
  function getDeviceInfo() {
    return {
      ua: navigator.userAgent || "",
      language: navigator.language || "",
      platform: navigator.platform || ""
    };
  }

  // Prepare payload and send to /monitor_activity
  async function sendBehaviorData() {
    const userElem = document.getElementById("user") || document.getElementById("username") || {};
    const user_id = userElem.value || userElem.getAttribute && userElem.getAttribute("value") || "guest";

    // Build frame_data placeholder (frontend cannot send full video frames here).
    // If you later add backend webcam/video processing, include numeric scores in frame_data.
    const frame_data = {
      // optional placeholders: frame_trust, gaze_score, pose_score
      frame_trust: null,
      gaze_score: null,
      pose_score: null
    };

    const payload = {
      user_id: user_id,
      keystrokes: keystrokes,
      frame_data: frame_data,
      device_info: getDeviceInfo()
    };

    try {
      const res = await fetch("/monitor_activity", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        console.error("monitor_activity returned", res.status);
        return;
      }

      const data = await res.json();

      // Update UI - ensure these elements exist in your dashboard/index templates
      const trustEl = document.getElementById("trust") || document.getElementById("trust_score");
      const riskEl = document.getElementById("risk") || document.getElementById("risk_status");

      if (trustEl) trustEl.innerText = (typeof data.trust_score !== "undefined") ? data.trust_score : "N/A";
      if (riskEl) riskEl.innerText = data.risk || (data.anomaly ? "high" : "normal");

      // After successful send, clear buffer
      keystrokes = [];
    } catch (err) {
      console.error("Error sending behavior data:", err);
    }
  }

  // Send every N seconds
  const INTERVAL_MS = 5000; // 5 seconds
  setInterval(sendBehaviorData, INTERVAL_MS);

  // Expose for debugging
  window.__brocode_monitor = {
    getBuffer: () => keystrokes.slice(),
    flushNow: sendBehaviorData
  };
})();

async function initWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    const video = document.createElement("video");
    video.srcObject = stream;
    video.play();
    // Simulate minor gaze variation
    setInterval(() => {
      gazeScore = 0.6 + 0.2 * Math.random();
      poseScore = 0.6 + 0.3 * Math.random();
    }, 3000);
  } catch (err) {
    console.warn("Webcam not available or permission denied:", err);
  }
}
initWebcam();

// Later in sendBehaviorData()
const frame_data = {
  gaze_score: gazeScore,
  pose_score: poseScore,
  frame_trust: (gazeScore + poseScore) / 2
};