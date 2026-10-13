import os
import time
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyautogui

# --- CONFIGURATION CONSTANTS ---
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"
CAMERA_INDEX = 0
CLICK_COOLDOWN = 0.35      
PINCH_THRESHOLD = 30       
EDGE_MARGIN = 100          

# --- ONE-EURO FILTER FOR BUTTERY-SMOOTH MOTION ---
class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def smoothing_factor(self, t, cutoff):
        r = 2 * np.pi * cutoff * (t - self.t_prev)
        return r / (r + 1)

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0.0:
            return self.x_prev
        
        a_d = self.smoothing_factor(t, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self.smoothing_factor(t, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

def download_model_if_missing(path, url):
    if not os.path.exists(path):
        print("[INFO] Downloading MediaPipe Hand Landmarker model...")
        try:
            urllib.request.urlretrieve(url, path)
            print("[INFO] Model downloaded successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to download model: {e}")
            exit(1)

def draw_hand_skeleton(frame, landmarks, w, h):
    """Draws MediaPipe hand skeleton connections and joints."""
    connections = [
        (0,1),(1,2),(2,3),(3,4),         # Thumb
        (0,5),(5,6),(6,7),(7,8),         # Index
        (5,9),(9,10),(10,11),(11,12),    # Middle
        (9,13),(13,14),(14,15),(15,16),  # Ring
        (13,17),(17,18),(18,19),(19,20), # Pinky
        (0,17)                           # Palm base
    ]
    
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    
    # Draw connections
    for start, end in connections:
        cv2.line(frame, pts[start], pts[end], (200, 200, 200), 2)
        
    # Draw joint dots
    for pt in pts:
        cv2.circle(frame, pt, 5, (0, 255, 0), cv2.FILLED)

def main():
    pyautogui.FAILSAFE = False
    download_model_if_missing(MODEL_PATH, MODEL_URL)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options, 
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO
    )
    detector = vision.HandLandmarker.create_from_options(options)

    screen_w, screen_h = pyautogui.size()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FPS, 60)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    init_time = time.time()
    filter_x = OneEuroFilter(init_time, screen_w / 2, min_cutoff=0.8, beta=0.02)
    filter_y = OneEuroFilter(init_time, screen_h / 2, min_cutoff=0.8, beta=0.02)
    filter_scroll = OneEuroFilter(init_time, 0.0, min_cutoff=1.0, beta=0.05)

    last_action_time = 0
    scroll_accumulator = 0
    prev_scroll_y_smooth = None

    print("[INFO] AirTouch Virtual Mouse is running. Press 'q' to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        # Draw AirTouch Tracking Region Box
        cv2.rectangle(frame, (EDGE_MARGIN, EDGE_MARGIN), (w - EDGE_MARGIN, h - EDGE_MARGIN), (100, 100, 100), 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        current_time = time.time()
        frame_timestamp_ms = int((current_time - init_time) * 1000)
        detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

        # HUD State Values
        tracking_status = "INACTIVE"
        gesture_text = "NONE"
        status_text = "IDLE"
        action_text = "PAUSED"
        action_color = (0, 0, 255)

        if detection_result.hand_landmarks:
            tracking_status = "ACTIVE"
            landmarks = detection_result.hand_landmarks[0]
            draw_hand_skeleton(frame, landmarks, w, h)

            thumb_x, thumb_y = int(landmarks[4].x * w), int(landmarks[4].y * h)
            index_x, index_y = int(landmarks[8].x * w), int(landmarks[8].y * h)
            middle_x, middle_y = int(landmarks[12].x * w), int(landmarks[12].y * h)
            ring_x, ring_y = int(landmarks[16].x * w), int(landmarks[16].y * h)

            # Finger extension checks
            index_extended = landmarks[8].y < landmarks[6].y
            middle_extended = landmarks[12].y < landmarks[10].y
            ring_extended = landmarks[16].y < landmarks[14].y
            pinky_extended = landmarks[20].y < landmarks[18].y

            distance_left = np.hypot(index_x - thumb_x, index_y - thumb_y)
            distance_right = np.hypot(middle_x - thumb_x, middle_y - thumb_y)
            distance_double = np.hypot(ring_x - thumb_x, ring_y - thumb_y)

            # Count total extended fingers for open palm / fist detection
            extended_count = sum([index_extended, middle_extended, ring_extended, pinky_extended])

            if extended_count == 0:
                gesture_text = "FIST"
                status_text = "IDLE"
                action_text = "PAUSED"
                action_color = (0, 0, 255)
                prev_scroll_y_smooth = None

            elif extended_count >= 4:
                gesture_text = "OPEN PALM"
                status_text = "HOVER"
                action_text = "PAUSED"
                action_color = (0, 165, 255)
                prev_scroll_y_smooth = None

            # 1. SCROLL MODE: Index and Middle extended
            elif index_extended and middle_extended and not ring_extended and not pinky_extended:
                gesture_text = "2 FINGERS"
                status_text = "SCROLLING"
                action_text = "SCROLL"
                action_color = (0, 255, 255)

                smooth_scroll_y = filter_scroll(current_time, index_y)
                if prev_scroll_y_smooth is not None:
                    diff_y = smooth_scroll_y - prev_scroll_y_smooth
                    scroll_accumulator += diff_y
                    scroll_step = 12
                    if abs(scroll_accumulator) >= scroll_step:
                        clicks = int(scroll_accumulator / scroll_step)
                        pyautogui.scroll(int(-clicks * 40))
                        scroll_accumulator = 0
                prev_scroll_y_smooth = smooth_scroll_y

            else:
                prev_scroll_y_smooth = None
                scroll_accumulator = 0

                # 2. DOUBLE CLICK: Ring Finger Pinch
                if distance_double < PINCH_THRESHOLD:
                    gesture_text = "PINCH (RING)"
                    status_text = "ACTIVE"
                    action_text = "DOUBLE CLICK"
                    action_color = (255, 0, 255)
                    if current_time - last_action_time > CLICK_COOLDOWN:
                        pyautogui.doubleClick()
                        last_action_time = current_time

                # 3. RIGHT CLICK: Middle Finger Pinch
                elif distance_right < PINCH_THRESHOLD and distance_left >= PINCH_THRESHOLD:
                    gesture_text = "3 FINGERS"
                    status_text = "ACTIVE"
                    action_text = "RIGHT CLICK"
                    action_color = (0, 0, 255)
                    if current_time - last_action_time > CLICK_COOLDOWN:
                        pyautogui.rightClick()
                        last_action_time = current_time

                # 4. LEFT CLICK: Index Finger Pinch
                elif distance_left < PINCH_THRESHOLD:
                    gesture_text = "PINCH (INDEX)"
                    status_text = "ACTIVE"
                    action_text = "LEFT CLICK"
                    action_color = (0, 255, 0)
                    if current_time - last_action_time > CLICK_COOLDOWN:
                        pyautogui.click()
                        last_action_time = current_time

                # 5. CURSOR MOVEMENT
                else:
                    gesture_text = "POINT"
                    status_text = "ACTIVE"
                    action_text = "MOVE CURSOR"
                    action_color = (255, 0, 0)

                    raw_target_x = np.interp(index_x, (EDGE_MARGIN, w - EDGE_MARGIN), (0, screen_w))
                    raw_target_y = np.interp(index_y, (EDGE_MARGIN, h - EDGE_MARGIN), (0, screen_h))

                    smooth_x = filter_x(current_time, raw_target_x)
                    smooth_y = filter_y(current_time, raw_target_y)

                    pyautogui.moveTo(smooth_x, smooth_y)

        # --- DRAW AIRTOUCH HUD PANELS ON VIDEO FEED ---
        # Left Panel (HUD Status)
        cv2.rectangle(frame, (10, 10), (240, 150), (30, 30, 30), cv2.FILLED)
        cv2.putText(frame, "AIRTOUCH", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Tracking: {tracking_status}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if tracking_status=="ACTIVE" else (0, 0, 255), 1)
        cv2.putText(frame, f"Gesture:  {gesture_text}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
        cv2.putText(frame, f"Status:   {status_text}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Action:   {action_text}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, action_color, 1)

        # Right Panel (Gesture Legend)
        cv2.rectangle(frame, (w - 240, 10), (w - 10, 170), (30, 30, 30), cv2.FILLED)
        cv2.putText(frame, "GESTURE LEGEND", (w - 230, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        legends = [
            ("POINT", "Move / Scroll"),
            ("PINCH", "Left Click"),
            ("3 FINGERS", "Right Click"),
            ("2 FINGERS", "Double Click"),
            ("OPEN PALM", "Pause"),
            ("FIST", "Idle")
        ]
        for i, (gest, act) in enumerate(legends):
            cv2.putText(frame, f"{gest}", (w - 230, 55 + (i * 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(frame, f"-> {act}", (w - 120, 55 + (i * 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)

        cv2.imshow("AirTouch - Touchless Mouse Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()