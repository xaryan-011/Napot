"""
NapNot - Sleep Detection System (Consolidated Single File)
==========================================================
Real-time driver drowsiness detection using MediaPipe Face Mesh,
Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and multi-stage alarms.
"""

import time
import threading
import subprocess
import cv2
import numpy as np
import mediapipe as mp
import pygame

# --- CONFIGURATION CONSTANTS ---
EAR_THRESHOLD_DEFAULT = 0.25      # Fallback threshold if calibration fails
EAR_CALIBRATION_MULTIPLIER = 0.75 # Threshold = baseline_average_ear * multiplier
CALIBRATION_DURATION_SEC = 3.0    # Time (seconds) to collect baseline open-eye EAR

MAR_THRESHOLD = 0.50             # MAR threshold to consider mouth open
YAWN_DURATION_SEC = 2.0          # Time (seconds) mouth must stay open to log a yawn

ALARM_STAGE1_SEC = 2.0            # Trigger soft alarm (2 seconds closed)
ALARM_STAGE2_SEC = 4.0            # Trigger loud alarm (4 seconds closed)
ALARM_STAGE3_SEC = 6.0            # Trigger voice warning (6 seconds closed)

# MediaPipe landmark indices for eye and mouth contours
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_CORNER_LEFT = 78
MOUTH_CORNER_RIGHT = 308
MOUTH_VERT_PAIRS = [(82, 87), (13, 14), (312, 317)]

# --- THREADED CAMERA CLASS ---
class ThreadedCamera:
    """Spawns a separate thread to read frames from the webcam to avoid UI lag."""
    def __init__(self, src=0, width=640, height=480):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        self.grabbed, self.frame = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        while self.started:
            grabbed, frame = self.stream.read()
            with self.read_lock:
                self.grabbed = grabbed
                if grabbed:
                    self.frame = frame
            time.sleep(0.01)

    def read(self):
        with self.read_lock:
            frame_copy = self.frame.copy() if self.frame is not None else None
            return self.grabbed, frame_copy

    def stop(self):
        self.started = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.stream.release()


# --- MATHEMATICAL FUNCTIONS ---
def distance(p1, p2):
    """Calculates Euclidean distance between two 3D/2D points."""
    return np.linalg.norm(p1 - p2)

def calculate_single_eye_ear(eye_points):
    """Computes EAR for a single eye contour."""
    # Vertical distances between upper and lower eyelids
    a = distance(eye_points[1], eye_points[5])
    b = distance(eye_points[2], eye_points[4])
    # Horizontal distance between outer and inner corners
    c = distance(eye_points[0], eye_points[3])
    return (a + b) / (2.0 * c) if c > 0 else 0.0

def calculate_ear(landmarks):
    """Extracts eye landmarks and returns the average EAR of both eyes."""
    left_pts = landmarks[LEFT_EYE_INDICES]
    right_pts = landmarks[RIGHT_EYE_INDICES]
    left_ear = calculate_single_eye_ear(left_pts)
    right_ear = calculate_single_eye_ear(right_pts)
    return (left_ear + right_ear) / 2.0, left_pts, right_pts

def calculate_mar(landmarks):
    """Calculates Mouth Aspect Ratio (MAR)."""
    vert_dists = [distance(landmarks[top], landmarks[bot]) for top, bot in MOUTH_VERT_PAIRS]
    horiz_dist = distance(landmarks[MOUTH_CORNER_LEFT], landmarks[MOUTH_CORNER_RIGHT])
    mar = sum(vert_dists) / (3.0 * horiz_dist) if horiz_dist > 0 else 0.0
    
    mouth_pts = [landmarks[MOUTH_CORNER_LEFT], landmarks[MOUTH_CORNER_RIGHT]]
    for top, bot in MOUTH_VERT_PAIRS:
        mouth_pts.extend([landmarks[top], landmarks[bot]])
    return mar, mouth_pts


# --- MULTI-STAGE ALARM SYSTEM ---
class AlarmSystem:
    """Manages audio warning triggers and handles text-to-speech warnings."""
    def __init__(self):
        pygame.mixer.init()
        # Load pre-existing WAV sounds (falls back to silence/warning print if not found)
        try:
            self.soft_alarm = pygame.mixer.Sound("alarm_soft.wav")
            self.loud_alarm = pygame.mixer.Sound("alarm_loud.wav")
        except Exception as e:
            print(f"[WARNING] Could not load WAV alarm files: {e}")
            self.soft_alarm = None
            self.loud_alarm = None
            
        self.active_stage = 0
        self.voice_speaking = False
        self.lock = threading.Lock()

    def _speak_worker(self, message):
        try:
            # PowerShell SAPI Speech Synthesis (native on Windows)
            cmd = (
                "Add-Type -AssemblyName System.Speech; "
                f"$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$speak.Speak('{message}')"
            )
            subprocess.run(["powershell", "-Command", cmd], capture_output=True)
        finally:
            with self.lock:
                self.voice_speaking = False

    def trigger_voice_alert(self, message):
        with self.lock:
            if not self.voice_speaking:
                self.voice_speaking = True
                t = threading.Thread(target=self._speak_worker, args=(message,), daemon=True)
                t.start()

    def update_alarms(self, closed_duration):
        if closed_duration >= ALARM_STAGE3_SEC:
            if self.active_stage != 3:
                self.stop_sounds()
                self.active_stage = 3
                print("[ALARM STAGE 3] VOICE WARNING ACTIVE")
            self.trigger_voice_alert("Warning! Driver drowsiness detected. Please wake up and pull over immediately.")
            
        elif closed_duration >= ALARM_STAGE2_SEC:
            if self.active_stage != 2:
                self.stop_sounds()
                self.active_stage = 2
                if self.loud_alarm:
                    self.loud_alarm.play(loops=-1)
                print("[ALARM STAGE 2] LOUD SIREN ACTIVE")
                
        elif closed_duration >= ALARM_STAGE1_SEC:
            if self.active_stage != 1:
                self.stop_sounds()
                self.active_stage = 1
                if self.soft_alarm:
                    self.soft_alarm.play(loops=-1)
                print("[ALARM STAGE 1] SOFT BEEP ACTIVE")
                
        else:
            if self.active_stage != 0:
                self.stop_sounds()
                self.active_stage = 0

    def stop_sounds(self):
        if self.soft_alarm:
            self.soft_alarm.stop()
        if self.loud_alarm:
            self.loud_alarm.stop()

    def close(self):
        self.stop_sounds()
        pygame.mixer.quit()


# --- MAIN APPLICATION ENTRY POINT ---
def main():
    print("=" * 60)
    print("  NapNot - Simplified Real-Time Drowsiness Detector")
    print("=" * 60)
    
    # Initialize Pygame audio and components
    alarm_system = AlarmSystem()
    
    # Initialize Camera
    camera = ThreadedCamera(src=0).start()
    
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Core driver metrics
    is_calibrated = False
    calibration_vals = []
    calibrated_threshold = EAR_THRESHOLD_DEFAULT
    
    total_blinks = 0
    total_yawns = 0
    
    yawn_active = False
    yawn_start_time = None
    
    eye_closed_start_time = None
    last_eye_closed = False
    
    calibration_start_time = None
    
    fps = 30.0
    prev_time = time.time()
    
    print("[OK] Initialized camera and models. Starting tracking loop...")
    print("Press 'q' in the window to quit, or 'r' to reset counters.")
    
    try:
        while True:
            grabbed, raw_frame = camera.read()
            if not grabbed or raw_frame is None:
                time.sleep(0.01)
                continue
            
            # Flip horizontally for mirrored feel
            frame = cv2.flip(raw_frame, 1)
            h, w = frame.shape[:2]
            
            # Calculate FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            prev_time = curr_time
            
            # Process Frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            closed_duration = 0.0
            status_text = "No Face Detected"
            status_color = (0, 0, 255) # Red for no face
            
            # Initialize metrics for display fallback
            ear = 0.0
            mar = 0.0
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Convert landmarks to numpy coordinate array
                coords = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in face_landmarks.landmark], dtype=np.float32)
                
                # Draw Face Bounding Box
                x_min, y_min = np.min(coords[:, :2], axis=0)
                x_max, y_max = np.max(coords[:, :2], axis=0)
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (200, 200, 200), 1)
                
                # Compute EAR and MAR
                ear, left_eye, right_eye = calculate_ear(coords)
                mar, mouth_pts = calculate_mar(coords)
                
                # Eye and Mouth Landmark Drawing
                cv2.polylines(frame, [left_eye[:, :2].astype(int)], True, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.polylines(frame, [right_eye[:, :2].astype(int)], True, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.polylines(frame, [np.array(mouth_pts)[:, :2].astype(int)], True, (255, 255, 0), 1, cv2.LINE_AA)
                
                # Dynamic Calibration (first 3 seconds after face is seen)
                if not is_calibrated:
                    if calibration_start_time is None:
                        calibration_start_time = time.time()
                    
                    elapsed_calib = time.time() - calibration_start_time
                    if elapsed_calib < CALIBRATION_DURATION_SEC:
                        calibration_vals.append(ear)
                        status_text = f"CALIBRATING ({int((elapsed_calib/CALIBRATION_DURATION_SEC)*100)}%)"
                        status_color = (255, 255, 0) # Cyan/Yellow
                    else:
                        if len(calibration_vals) > 10:
                            avg_ear = np.mean(calibration_vals)
                            if avg_ear > 0.15:
                                calibrated_threshold = avg_ear * EAR_CALIBRATION_MULTIPLIER
                                is_calibrated = True
                                print(f"[OK] Calibration complete! Baseline EAR: {avg_ear:.3f}, Threshold: {calibrated_threshold:.3f}")
                            else:
                                # Re-run calibration if average is bad (eyes were likely closed)
                                calibration_vals = []
                                calibration_start_time = time.time()
                else:
                    # Tracking Active
                    is_closed = ear < calibrated_threshold
                    
                    # Eye state change tracking
                    if is_closed and not last_eye_closed:
                        eye_closed_start_time = time.time()
                        last_eye_closed = True
                    elif not is_closed and last_eye_closed:
                        # Eye opened
                        if eye_closed_start_time is not None:
                            duration = time.time() - eye_closed_start_time
                            if duration >= 0.08: # Minimum duration to register as blink
                                total_blinks += 1
                        last_eye_closed = False
                        eye_closed_start_time = None
                    
                    if is_closed and eye_closed_start_time is not None:
                        closed_duration = time.time() - eye_closed_start_time
                    
                    # Yawn logic
                    if mar > MAR_THRESHOLD:
                        if not yawn_active:
                            yawn_active = True
                            yawn_start_time = time.time()
                    else:
                        if yawn_active:
                            yawn_duration = time.time() - yawn_start_time
                            if yawn_duration >= YAWN_DURATION_SEC:
                                total_yawns += 1
                            yawn_active = False
                            yawn_start_time = None
                    
                    # Determine fatigue state
                    if closed_duration >= ALARM_STAGE2_SEC:
                        status_text = "SLEEPING"
                        status_color = (0, 0, 255) # Red
                    elif closed_duration >= ALARM_STAGE1_SEC:
                        status_text = "DROWSY"
                        status_color = (0, 165, 255) # Orange
                    else:
                        status_text = "NORMAL / ALERT"
                        status_color = (0, 255, 0) # Green
            
            # Update Alarm System state
            alarm_system.update_alarms(closed_duration)
            
            # HUD overlay (simple text directly on camera frame)
            # Semi-transparent backdrop for text readability
            cv2.rectangle(frame, (10, 10), (280, 160), (20, 20, 20), -1)
            cv2.rectangle(frame, (10, 10), (280, 160), (60, 60, 60), 1)
            
            cv2.putText(frame, "NapNot Sleep Detector", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, f"EAR: {ear:.3f} / {calibrated_threshold:.3f}" if results.multi_face_landmarks else "EAR: N/A", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, f"MAR: {mar:.3f}" if results.multi_face_landmarks else "MAR: N/A", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Blinks: {total_blinks} | Yawns: {total_yawns}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            
            # Status Text
            cv2.putText(frame, f"STATE: {status_text}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2, cv2.LINE_AA)
            
            # Alert banner if closed_duration is active
            if closed_duration > 0:
                cv2.putText(frame, f"EYES CLOSED: {closed_duration:.1f}s", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                
            # Show display window
            cv2.imshow("NapNot Drowsiness Monitor", frame)
            
            # Handle Keyboard Input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                total_blinks = 0
                total_yawns = 0
                calibration_vals = []
                is_calibrated = False
                calibration_start_time = None
                alarm_system.stop_sounds()
                print("[OK] Reset all counters and recalibrating...")
                
    finally:
        # Cleanup
        print("[*] Releasing resources and shutting down...")
        camera.stop()
        face_mesh.close()
        alarm_system.close()
        cv2.destroyAllWindows()
        print("[OK] Exit clean.")

if __name__ == "__main__":
    main()
