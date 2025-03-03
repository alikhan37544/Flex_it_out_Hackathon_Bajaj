import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import random
import math

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
# Power Pose Class
# ----------------------------
class PowerPose:
    def __init__(self, name, description, landmark_criteria, bonus_score=100, duration=5.0):
        self.name = name
        self.description = description
        self.landmark_criteria = landmark_criteria  # Function that takes landmarks and returns True/False
        self.bonus_score = bonus_score
        self.duration = duration  # How long the pose challenge is active
        self.progress = 0.0       # Progress toward completing the pose (0.0-1.0)
        self.hold_time = 0.0      # How long the player has been in the correct pose
        self.required_hold_time = 1.5  # Seconds player must hold pose to get bonus
        self.completed = False
        self.active_start_time = 0.0

    def reset(self, current_time):
        self.progress = 0.0
        self.hold_time = 0.0
        self.completed = False
        self.active_start_time = current_time

    def is_expired(self, current_time):
        return current_time - self.active_start_time > self.duration

    def update(self, landmarks, dt):
        if self.completed:
            return False
        
        is_in_pose = self.landmark_criteria(landmarks)
        
        if is_in_pose:
            self.hold_time += dt
            self.progress = min(1.0, self.hold_time / self.required_hold_time)
            if self.hold_time >= self.required_hold_time:
                self.completed = True
                return True  # Pose was just completed
        else:
            # Reset hold time if pose is broken
            self.hold_time = max(0, self.hold_time - dt * 2)  # Decay faster than buildup
            self.progress = min(1.0, self.hold_time / self.required_hold_time)
            
        return False

# ----------------------------
# Visual Effects Class
# ----------------------------
class VisualEffect:
    def __init__(self, effect_type, duration=1.0, intensity=1.0, start_time=0.0):
        self.effect_type = effect_type  # "flash", "particles", "zoom", etc.
        self.duration = duration
        self.intensity = intensity
        self.start_time = start_time
        self.elapsed_time = 0.0
        self.particles = []
        if effect_type == "particles":
            self.generate_particles()

    def generate_particles(self, count=50):
        for _ in range(count):
            self.particles.append({
                'x': random.randint(0, 640),
                'y': random.randint(0, 480),
                'vx': random.uniform(-5, 5),
                'vy': random.uniform(-8, -2),
                'size': random.randint(5, 15),
                'color': (
                    random.randint(150, 255),
                    random.randint(150, 255),
                    random.randint(150, 255)
                ),
                'life': random.uniform(0.5, 1.5)
            })

    def update(self, dt):
        self.elapsed_time += dt
        
        if self.effect_type == "particles":
            for p in self.particles[:]:
                p['x'] += p['vx']
                p['y'] += p['vy']
                p['vy'] += 0.1  # Gravity
                p['life'] -= dt
                
                if p['life'] <= 0:
                    self.particles.remove(p)
                    
        return self.elapsed_time < self.duration

    def render(self, frame):
        if self.effect_type == "flash":
            # Apply a brightness boost that fades over time
            alpha = 1.0 - (self.elapsed_time / self.duration)
            overlay = np.ones_like(frame) * 255
            cv2.addWeighted(overlay, alpha * self.intensity * 0.3, frame, 1, 0, frame)
            
        elif self.effect_type == "particles":
            for p in self.particles:
                alpha = min(1.0, p['life'])
                size = int(p['size'] * alpha)
                color = tuple([int(c * alpha) for c in p['color']])
                cv2.circle(frame, (int(p['x']), int(p['y'])), size, color, -1)
                
        elif self.effect_type == "vignette":
            h, w = frame.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            center = (w // 2, h // 2)
            
            # Create radial gradient
            for y in range(h):
                for x in range(w):
                    distance = math.sqrt((x - center[0])**2 + (y - center[1])**2)
                    max_distance = math.sqrt(center[0]**2 + center[1]**2)
                    # Pulsing effect based on elapsed time
                    pulsation = 0.2 * math.sin(self.elapsed_time * 10)
                    norm_distance = (distance / max_distance) + pulsation
                    mask[y, x] = max(0, min(255, int(255 * (1 - norm_distance))))
            
            # Apply the mask
            alpha = 1.0 - (self.elapsed_time / self.duration) * 0.7
            color_overlay = np.zeros_like(frame)
            color_overlay[:,:] = (0, 0, 255)  # Blue overlay
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            weighted_mask = mask_3d * alpha * self.intensity
            frame = cv2.addWeighted(frame, 1.0, color_overlay, 1.0, 0)
            frame = (frame * weighted_mask + frame * (1 - weighted_mask)).astype(np.uint8)
            
        return frame

# ----------------------------
# Power Pose Definitions
# ----------------------------
def define_power_poses():
    poses = []
    
    # T-Pose: Arms stretched out horizontally
    def t_pose_criteria(landmarks):
        if not landmarks:
            return False
            
        # Get relevant landmarks
        left_shoulder = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]
        left_elbow = landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value]
        left_wrist = landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value]
        
        # Check if arms are stretched horizontally
        # Shoulders should be roughly at the same height
        if abs(left_shoulder.y - right_shoulder.y) > 0.05:
            return False
            
        # Elbows should be roughly at the same height as shoulders
        if abs(left_elbow.y - left_shoulder.y) > 0.05 or abs(right_elbow.y - right_shoulder.y) > 0.05:
            return False
            
        # Wrists should be roughly at the same height as elbows
        if abs(left_wrist.y - left_elbow.y) > 0.05 or abs(right_wrist.y - right_elbow.y) > 0.05:
            return False
            
        # Arms should be stretched out (wrists should be far from shoulders horizontally)
        if (left_wrist.x >= left_shoulder.x) or (right_wrist.x <= right_shoulder.x):
            return False
            
        return True
        
    poses.append(PowerPose(
        "T-Pose", 
        "Stretch your arms horizontally like a T",
        t_pose_criteria,
        bonus_score=150,
        duration=8.0
    ))
    
    # Disco Fever: One arm up, one arm down
    def disco_fever_criteria(landmarks):
        if not landmarks:
            return False
            
        left_shoulder = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]
        left_elbow = landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value]
        left_wrist = landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value]
        
        # Check if right arm is raised (wrist above shoulder)
        right_arm_up = right_wrist.y < right_shoulder.y - 0.1
        
        # Check if left arm is pointing down (wrist below shoulder)
        left_arm_down = left_wrist.y > left_shoulder.y + 0.1
        
        return right_arm_up and left_arm_down
        
    poses.append(PowerPose(
        "Disco Fever", 
        "Right arm up, left arm down",
        disco_fever_criteria,
        bonus_score=120,
        duration=7.0
    ))
    
    # Superhero: Hands on hips
    def superhero_criteria(landmarks):
        if not landmarks:
            return False
            
        left_shoulder = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]
        left_elbow = landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value]
        left_wrist = landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value]
        left_hip = landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value]
        
        # Arms should be bent (elbows away from body)
        elbows_out = (left_elbow.x < left_shoulder.x) and (right_elbow.x > right_shoulder.x)
        
        # Wrists should be near hips
        hands_on_hips = (
            abs(left_wrist.y - left_hip.y) < 0.08 and
            abs(right_wrist.y - right_hip.y) < 0.08 and
            abs(left_wrist.x - left_hip.x) < 0.1 and
            abs(right_wrist.x - right_hip.x) < 0.1
        )
        
        return elbows_out and hands_on_hips
        
    poses.append(PowerPose(
        "Superhero", 
        "Stand tall with hands on hips",
        superhero_criteria,
        bonus_score=180,
        duration=7.5
    ))
    
    # Dab: One arm extended diagonally up, other arm folded across body
    def dab_criteria(landmarks):
        if not landmarks:
            return False
            
        left_shoulder = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]
        left_elbow = landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value]
        left_wrist = landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value]
        nose = landmarks[mp_holistic.PoseLandmark.NOSE.value]
        
        # Left arm extended up diagonally
        left_arm_up = (left_wrist.y < left_elbow.y) and (left_elbow.y < left_shoulder.y)
        left_arm_diagonal = left_wrist.x < left_elbow.x < left_shoulder.x
        
        # Right arm across body, with elbow bent
        right_arm_across = right_wrist.x < right_elbow.x
        
        # Right elbow near or above face level
        right_elbow_up = right_elbow.y < right_shoulder.y
        
        # Face turned toward right arm
        face_turned = nose.x < (left_shoulder.x + right_shoulder.x) / 2
        
        return left_arm_up and left_arm_diagonal and right_arm_across and right_elbow_up
        
    poses.append(PowerPose(
        "Dab", 
        "Left arm up diagonal, right arm folded across",
        dab_criteria,
        bonus_score=200,
        duration=6.0
    ))
    
    return poses

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

# Power pose settings
power_poses = define_power_poses()
active_power_pose = None
power_pose_interval_min = 15.0  # Min seconds between power pose challenges
power_pose_interval_max = 30.0  # Max seconds between power pose challenges
next_power_pose_time = time.time() + random.uniform(power_pose_interval_min, power_pose_interval_max)
pose_reference_images = {}  # Will store reference images for poses

# Visual effects
visual_effects = []

# For delta time calculation
last_frame_time = time.time()

# ----------------------------
# Load or create pose reference images
# ----------------------------
def create_pose_references():
    poses = {}
    
    # T-Pose reference
    img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    cv2.line(img, (150, 150), (50, 150), (0, 0, 0), 5)  # Left arm
    cv2.line(img, (150, 150), (250, 150), (0, 0, 0), 5)  # Right arm
    cv2.line(img, (150, 150), (150, 250), (0, 0, 0), 5)  # Body
    cv2.circle(img, (150, 100), 30, (0, 0, 0), 2)  # Head
    poses["T-Pose"] = img
    
    # Disco Fever reference
    img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    cv2.line(img, (150, 150), (50, 200), (0, 0, 0), 5)  # Left arm down
    cv2.line(img, (150, 150), (250, 100), (0, 0, 0), 5)  # Right arm up
    cv2.line(img, (150, 150), (150, 250), (0, 0, 0), 5)  # Body
    cv2.circle(img, (150, 100), 30, (0, 0, 0), 2)  # Head
    poses["Disco Fever"] = img
    
    # Superhero reference
    img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    cv2.line(img, (150, 150), (110, 200), (0, 0, 0), 5)  # Left arm bent
    cv2.line(img, (150, 150), (190, 200), (0, 0, 0), 5)  # Right arm bent
    cv2.line(img, (150, 150), (150, 250), (0, 0, 0), 5)  # Body
    cv2.circle(img, (150, 100), 30, (0, 0, 0), 2)  # Head
    # Highlight hands on hips
    cv2.circle(img, (110, 200), 10, (0, 0, 255), -1)  # Left hand
    cv2.circle(img, (190, 200), 10, (0, 0, 255), -1)  # Right hand
    poses["Superhero"] = img
    
    # Dab reference
    img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    cv2.line(img, (150, 150), (80, 80), (0, 0, 0), 5)  # Left arm up diagonal
    cv2.line(img, (150, 150), (170, 110), (0, 0, 0), 5)  # Right arm bent
    cv2.line(img, (170, 110), (120, 120), (0, 0, 0), 5)  # Right forearm across
    cv2.line(img, (150, 150), (150, 250), (0, 0, 0), 5)  # Body
    cv2.circle(img, (130, 100), 30, (0, 0, 0), 2)  # Head (slightly turned)
    poses["Dab"] = img
    
    return poses

pose_reference_images = create_pose_references()

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
        # Power Pose Management
        # ----------------------------
        # Check if it's time to trigger a new power pose challenge
        if active_power_pose is None and current_time >= next_power_pose_time:
            active_power_pose = random.choice(power_poses)
            active_power_pose.reset(current_time)
            # Add visual effect for pose challenge start
            visual_effects.append(VisualEffect("flash", duration=1.0, intensity=1.0, start_time=current_time))
            visual_effects.append(VisualEffect("vignette", duration=active_power_pose.duration, intensity=1.0, start_time=current_time))
        
        # Update active power pose if there is one
        if active_power_pose is not None:
            # Check if the pose duration has expired
            if active_power_pose.is_expired(current_time):
                active_power_pose = None
                next_power_pose_time = current_time + random.uniform(power_pose_interval_min, power_pose_interval_max)
            else:
                # Check if player is in the correct pose
                if results.pose_landmarks:
                    pose_landmarks = results.pose_landmarks.landmark
                    pose_completed = active_power_pose.update(pose_landmarks, dt)
                    
                    if pose_completed:
                        # Player successfully completed the pose!
                        score += active_power_pose.bonus_score
                        # Add celebratory visual effects
                        visual_effects.append(VisualEffect("flash", duration=1.0, intensity=1.5, start_time=current_time))
                        visual_effects.append(VisualEffect("particles", duration=3.0, intensity=1.0, start_time=current_time))
                        # Set up for next pose
                        active_power_pose = None
                        next_power_pose_time = current_time + random.uniform(power_pose_interval_min, power_pose_interval_max)
                
                # Draw pose information on screen
                if active_power_pose is not None:
                    # Create a semi-transparent overlay panel
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (w//2 - 150, 70), (w//2 + 150, 170), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                    
                    # Draw pose name and instructions
                    cv2.putText(frame, f"POWER POSE: {active_power_pose.name}", (w//2 - 140, 95),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, active_power_pose.description, (w//2 - 140, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Draw pose progress bar
                    bar_width = 280
                    filled_width = int(bar_width * active_power_pose.progress)
                    cv2.rectangle(frame, (w//2 - 140, 135), (w//2 - 140 + bar_width, 150), (100, 100, 100), -1)
                    cv2.rectangle(frame, (w//2 - 140, 135), (w//2 - 140 + filled_width, 150), (0, 255, 0), -1)
                    
                    # Display time remaining
                    time_left = max(0, active_power_pose.duration - (current_time - active_power_pose.active_start_time))
                    cv2.putText(frame, f"Time: {time_left:.1f}s", (w//2 + 70, 145),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Show reference image in corner
                    if active_power_pose.name in pose_reference_images:
                        ref_img = pose_reference_images[active_power_pose.name]
                        ref_h, ref_w = ref_img.shape[:2]
                        scale = 0.5
                        resized_ref = cv2.resize(ref_img, (int(ref_w * scale), int(ref_h * scale)))
                        ref_h, ref_w = resized_ref.shape[:2]
                        
                        # Create region of interest
                        roi = frame[10:10+ref_h, w-10-ref_w:w-10]
                        
                        # Create mask for smooth blending
                        mask = np.ones((ref_h, ref_w), dtype=np.uint8) * 255
                        
                        # Blend reference image onto frame
                        frame[10:10+ref_h, w-10-ref_w:w-10] = cv2.bitwise_and(
                            roi, roi, mask=cv2.bitwise_not(mask)) + cv2.bitwise_and(
                            resized_ref, resized_ref, mask=mask)

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
                        # Add a particle effect when collecting a multiplier
                        visual_effects.append(VisualEffect("particles", duration=1.0, intensity=0.7, start_time=current_time))
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
        # Update Visual Effects
        # ----------------------------
        # Update and render all active visual effects
        for effect in visual_effects[:]:
            is_active = effect.update(dt)
            if is_active:
                frame = effect.render(frame)
            else:
                visual_effects.remove(effect)

        # ----------------------------
        # UI Overlays
        # ----------------------------
        cv2.putText(frame, f"Score: {score:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Show Power Pose Bonus info if applicable
        if active_power_pose is not None and not active_power_pose.completed:
            bonus_text = f"Power Pose Bonus: +{active_power_pose.bonus_score}"
            text_size = cv2.getTextSize(bonus_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.putText(frame, bonus_text, (w - text_size[0] - 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Display music state
        if music_state == "dance":
            state_color = (0, 255, 0)  # Green for dance
            state_text = "Dance Dance Dance"
        else:
            state_color = (0, 0, 255)  # Red for freeze
            state_text = "Freeze Freeze Freeze"
            
        # Create a pulsing effect for the state text
        pulse_factor = 0.5 + 0.5 * abs(math.sin(current_time * 4))
        font_size = 1.0 + pulse_factor * 0.2
        thickness = 2 + int(pulse_factor * 2)
        
        # Draw the text with a black outline for better visibility
        text_size = cv2.getTextSize(state_text, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)[0]
        text_x = w // 2 - text_size[0] // 2
        text_y = 50
        
        # Draw outline
        cv2.putText(frame, state_text, (text_x-1, text_y-1),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), thickness+2)
        # Draw the colored text
        cv2.putText(frame, state_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, state_color, thickness)
        
        # Draw score progress bar
        max_score = 1000
        bar_width = int((score / max_score) * w)
        
        # Gradient bar background
        cv2.rectangle(frame, (0, h - 30), (w, h), (50, 50, 50), -1)
        
        # Animated gradient for progress
        for i in range(0, bar_width, 2):
            # Create a color gradient from blue to green to red
            ratio = i / w
            if ratio < 0.33:
                r = 0
                g = int(255 * (ratio / 0.33))
                b = 255
            elif ratio < 0.66:
                r = int(255 * ((ratio - 0.33) / 0.33))
                g = 255
                b = int(255 * (1 - ((ratio - 0.33) / 0.33)))
            else:
                r = 255
                g = int(255 * (1 - ((ratio - 0.66) / 0.34)))
                b = 0
                
            # Add time-based animation - convert to int to avoid the error
            wave = int(5 * math.sin(current_time * 5 + i * 0.1))
            y_pos = int(h - 30 + wave if i % 4 == 0 else h - 30)
            height = int(30 - wave if i % 4 == 0 else 30)
            
            cv2.rectangle(frame, (i, y_pos), (i + 2, h), (b, g, r), -1)
        
        # Bar outline
        cv2.rectangle(frame, (0, h - 30), (w, h), (255, 255, 255), 2)
        
        # Game instructions
        instructions = [
            "Wave over falling multipliers to boost score!",
            "DANCE when music plays, FREEZE when music stops!",
            "Strike POWER POSES for bonus points!"
        ]
        
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (10, h - 40 - 20 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # ----------------------------
        # Show Windows
        # ----------------------------
        # Apply a final subtle vignette effect to main display
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        
        # Create a soft radial gradient
        for y in range(h):
            for x in range(w):
                distance = math.sqrt((x - center[0])**2 + (y - center[1])**2)
                max_distance = math.sqrt(center[0]**2 + center[1]**2)
                norm_distance = distance / max_distance
                mask[y, x] = max(0, min(255, int(255 * (1 - norm_distance*0.5))))
        
        # Apply the mask for a subtle darkening effect at edges
        mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        frame = (frame * mask_3d).astype(np.uint8)
        
        # Show main game window
        cv2.imshow("Dance Dance Dance Game", frame)
        
        # Show smaller threshold window for debugging/feedback
        threshold_display = cv2.resize(threshold_colored, (320, 240))
        cv2.imshow("Movement Detection", threshold_display)

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