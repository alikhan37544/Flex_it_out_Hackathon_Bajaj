import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import random

# ----------------------------
# Multiplier Object Class
# ----------------------------
class Multiplier:
    def __init__(self, x, y, value, radius=20, speed=3, created_time=0.0):
        self.x = x
        self.y = y
        self.value = value      # e.g., 1.5, 2, 3
        self.radius = radius
        self.speed = speed      # falling speed (pixels per frame)
        self.collision_time = 0.0  # time the player has overlapped this token
        self.created_time = created_time  # for expiry checking

    def update(self):
        self.y += self.speed

# ----------------------------
# Initialization
# ----------------------------

# Initialize pygame mixer for music playback
pygame.mixer.init()
try:
    pygame.mixer.music.load("music.mp3")
except Exception as e:
    print("Error loading music file:", e)
    exit()

# We'll use a state variable for music:
# "dance" means music is playing, "freeze" means paused.
music_state = "dance"
# Start the music and immediately unpause to play from beginning
pygame.mixer.music.play(-1)
pygame.mixer.music.unpause()

# Set up our random durations:
dance_duration = random.uniform(20, 30)  # seconds for dance phase
freeze_duration = 5                      # seconds for freeze phase
next_state_change_time = time.time() + dance_duration

# Initialize MediaPipe Holistic for body tracking
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize webcam capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Read the first frame for motion detection
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame from webcam.")
    exit()
frame = cv2.resize(frame, (640, 480))
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

# Game variables
score = 0.0
score_scaling = 0.005       # How much score changes per movement unit
movement_threshold_value = 30  # For binarizing frame difference

# Multiplier settings
multipliers = []  # List of falling multiplier tokens
multiplier_spawn_interval = 3.0  # Seconds between spawns
last_multiplier_spawn_time = time.time()
collision_activation_time = 0.5    # Seconds required to activate a multiplier
active_multiplier_value = 1.0      # Currently active multiplier factor (default 1.0)
active_multiplier_timer = 0.0      # Remaining time (in seconds) for active multiplier
active_multiplier_duration = 5.0   # When activated, multiplier lasts for 5 seconds
token_expiry_duration = 7.0        # Tokens expire after 7 seconds if not collected

# For delta time calculation
last_frame_time = time.time()

# ----------------------------
# Main Loop
# ----------------------------
with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    while True:
        current_time = time.time()
        dt = current_time - last_frame_time
        last_frame_time = current_time

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 1)
        frame_copy = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with MediaPipe for holistic landmarks
        results = holistic.process(frame_rgb)

        # Draw MediaPipe landmarks for visual feedback
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.face_landmarks,
                connections=mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.pose_landmarks,
                connections=mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.left_hand_landmarks,
                connections=mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2))
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.right_hand_landmarks,
                connections=mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2))

        # ----------------------------
        # Compute Player Bounding Box from Pose Landmarks
        # ----------------------------
        bbox = None
        h, w, _ = frame.shape
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            xs = [int(lm.x * w) for lm in landmarks]
            ys = [int(lm.y * h) for lm in landmarks]
            x_min, x_max = max(min(xs) - 20, 0), min(max(xs) + 20, w)
            y_min, y_max = max(min(ys) - 20, 0), min(max(ys) + 20, h)
            bbox = (x_min, y_min, x_max, y_max)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # ----------------------------
        # Global Motion Detection (Frame Differencing)
        # ----------------------------
        gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        delta_frame = cv2.absdiff(prev_gray, gray)
        _, threshold_frame = cv2.threshold(delta_frame, movement_threshold_value, 255, cv2.THRESH_BINARY)
        threshold_frame = cv2.dilate(threshold_frame, None, iterations=2)
        prev_gray = gray.copy()

        # Create colored threshold image for visualization
        threshold_colored = cv2.cvtColor(threshold_frame, cv2.COLOR_GRAY2BGR)
        threshold_colored = cv2.applyColorMap(threshold_colored, cv2.COLORMAP_JET)

        # ----------------------------
        # Movement Analysis within Player ROI
        # ----------------------------
        movement = 0
        if bbox is not None:
            (x_min, y_min, x_max, y_max) = bbox
            roi_thresh = threshold_frame[y_min:y_max, x_min:x_max]
            movement = cv2.countNonZero(roi_thresh)
            # Blend threshold mask onto ROI for visual feedback
            roi_color = threshold_colored[y_min:y_max, x_min:x_max]
            overlay = frame[y_min:y_max, x_min:x_max]
            blended = cv2.addWeighted(overlay, 0.5, roi_color, 0.5, 0)
            frame[y_min:y_max, x_min:x_max] = blended

        # ----------------------------
        # Music State Management (Random Phases)
        # ----------------------------
        if current_time >= next_state_change_time:
            if music_state == "dance":
                pygame.mixer.music.pause()   # Pause the music (will resume later)
                music_state = "freeze"
                next_state_change_time = current_time + freeze_duration
            else:
                pygame.mixer.music.unpause() # Resume music from where it left off
                music_state = "dance"
                next_state_change_time = current_time + random.uniform(20, 30)

        # ----------------------------
        # Update and Spawn Multipliers
        # ----------------------------
        if current_time - last_multiplier_spawn_time > multiplier_spawn_interval:
            spawn_x = random.randint(30, w - 30)
            multiplier_value = random.choice([1.5, 2, 3])
            multipliers.append(Multiplier(spawn_x, -20, multiplier_value, radius=20,
                                          speed=random.randint(3, 6), created_time=current_time))
            last_multiplier_spawn_time = current_time

        # Update multiplier tokens
        for m in multipliers[:]:
            m.update()
            # Remove tokens that have fallen off or expired
            if m.y - m.radius > h or (current_time - m.created_time) > token_expiry_duration:
                multipliers.remove(m)
                continue

            # Check collision with player's bounding box
            if bbox is not None:
                if x_min <= m.x <= x_max and y_min <= m.y <= y_max:
                    sample_size = 20
                    sx = max(m.x - sample_size // 2, 0)
                    sy = max(m.y - sample_size // 2, 0)
                    ex = min(m.x + sample_size // 2, w)
                    ey = min(m.y + sample_size // 2, h)
                    roi_sample = threshold_frame[sy:ey, sx:ex]
                    motion_in_roi = cv2.countNonZero(roi_sample)
                    if motion_in_roi > 50:
                        m.collision_time += dt
                    else:
                        m.collision_time = 0.0

                    progress = min(int((m.collision_time / collision_activation_time) * 100), 100)
                    cv2.putText(frame, f"{progress}%", (m.x - 20, m.y - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    if m.collision_time >= collision_activation_time:
                        active_multiplier_value = m.value
                        active_multiplier_timer = active_multiplier_duration
                        multipliers.remove(m)
                else:
                    m.collision_time = 0.0

            # Draw the multiplier token
            color = (0, 255, 0)
            cv2.circle(frame, (m.x, int(m.y)), m.radius, color, 3)
            cv2.putText(frame, f"x{m.value}", (m.x - 15, int(m.y) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # ----------------------------
        # Manage Active Multiplier Timer
        # ----------------------------
        if active_multiplier_timer > 0:
            active_multiplier_timer -= dt
            cv2.putText(frame, f"Multiplier Active: x{active_multiplier_value} ({active_multiplier_timer:.1f}s)",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            active_multiplier_value = 1.0

        # ----------------------------
        # Update Score Based on Movement and Music Phase
        # ----------------------------
        # When in dance mode, score increases; in freeze mode, score decreases.
        if music_state == "dance":
            score += movement * score_scaling * active_multiplier_value
        else:
            score -= movement * score_scaling
        score = max(score, 0)

        # ----------------------------
        # UI Overlays
        # ----------------------------
        cv2.putText(frame, f"Score: {score:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if music_state == "dance":
            cv2.putText(frame, "Dance Dance Dance", (w//2 - 150, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "Freeze Freeze Freeze", (w//2 - 150, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        max_score = 1000
        bar_width = int((score / max_score) * w)
        cv2.rectangle(frame, (0, h - 30), (bar_width, h), (0, 255, 0), -1)
        cv2.rectangle(frame, (0, h - 30), (w, h), (255, 255, 255), 2)
        cv2.putText(frame, "Wave over falling multipliers to boost score!", (10, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # ----------------------------
        # Show Windows
        # ----------------------------
        cv2.imshow("Dance Dance Dance Game", frame)
        cv2.imshow("Threshold Movement", threshold_colored)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break

# ----------------------------
# Cleanup
# ----------------------------
cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
pygame.quit()
