import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import random
import math
import threading
import queue
import concurrent.futures
from multiprocessing import cpu_count

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
        self.lock = threading.Lock()
        if effect_type == "particles":
            self.generate_particles()

    def generate_particles(self, count=50):
        with self.lock:
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
            with self.lock:
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
            with self.lock:
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
def define_power_poses(mp_holistic):
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

# ----------------------------
# Thread-safe objects for frame processing
# ----------------------------
class FrameProcessor:
    def __init__(self, mp_holistic):
        self.mp_holistic = mp_holistic
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._process_frames)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
    def _process_frames(self):
        while self.running:
            try:
                frame_data = self.frame_queue.get(timeout=0.1)
                if frame_data is None:
                    continue
                    
                frame_id, frame = frame_data
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.holistic.process(frame_rgb)
                
                self.result_queue.put((frame_id, results))
                self.frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in frame processing thread: {e}")
                
    def submit_frame(self, frame_id, frame):
        try:
            self.frame_queue.put((frame_id, frame), block=False)
            return True
        except queue.Full:
            return False
            
    def get_result(self, timeout=0.1):
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

# ----------------------------
# Thread-safe motion detection
# ----------------------------
class MotionDetector:
    def __init__(self, threshold_value=30):
        self.threshold_value = threshold_value
        self.prev_gray = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.running = False
        self.thread = None
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._process_motion)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
    def _process_motion(self):
        while self.running:
            try:
                frame_data = self.frame_queue.get(timeout=0.1)
                if frame_data is None:
                    continue
                    
                frame_id, frame = frame_data
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                
                if self.prev_gray is None:
                    self.prev_gray = gray.copy()
                    self.result_queue.put((frame_id, None, None))
                    self.frame_queue.task_done()
                    continue
                
                delta_frame = cv2.absdiff(self.prev_gray, gray)
                _, threshold_frame = cv2.threshold(delta_frame, self.threshold_value, 255, cv2.THRESH_BINARY)
                threshold_frame = cv2.dilate(threshold_frame, None, iterations=2)
                
                # Create colored threshold image for visualization
                threshold_colored = cv2.cvtColor(threshold_frame, cv2.COLOR_GRAY2BGR)
                threshold_colored = cv2.applyColorMap(threshold_colored, cv2.COLORMAP_JET)
                
                self.prev_gray = gray.copy()
                self.result_queue.put((frame_id, threshold_frame, threshold_colored))
                self.frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in motion detection thread: {e}")
                
    def submit_frame(self, frame_id, frame):
        try:
            self.frame_queue.put((frame_id, frame), block=False)
            return True
        except queue.Full:
            return False
            
    def get_result(self, timeout=0.1):
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

# ----------------------------
# Thread-safe visual effects processor
# ----------------------------
class EffectsProcessor:
    def __init__(self):
        self.effects_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.running = False
        self.thread = None
        self.effects = []
        self.lock = threading.Lock()
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._process_effects)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
    def add_effect(self, effect):
        with self.lock:
            self.effects.append(effect)
            
    def _process_effects(self):
        while self.running:
            try:
                frame_data = self.effects_queue.get(timeout=0.1)
                if frame_data is None:
                    continue
                    
                frame_id, frame, dt = frame_data
                result_frame = frame.copy()
                
                with self.lock:
                    # Update all effects
                    for effect in self.effects[:]:
                        is_active = effect.update(dt)
                        if is_active:
                            result_frame = effect.render(result_frame)
                        else:
                            self.effects.remove(effect)
                
                self.result_queue.put((frame_id, result_frame))
                self.effects_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in effects processing thread: {e}")
                
    def submit_frame(self, frame_id, frame, dt):
        try:
            self.effects_queue.put((frame_id, frame, dt), block=False)
            return True
        except queue.Full:
            return False
            
    def get_result(self, timeout=0.1):
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def get_effects_count(self):
        with self.lock:
            return len(self.effects)

# ----------------------------
# Main function
# ----------------------------
def main():
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

    # Game variables
    score = 0.0
    score_scaling = 0.5       # How much score changes per movement unit
    movement_threshold_value = 10  # For binarizing frame difference

    # Multiplier settings
    multipliers = []  # List of falling multiplier tokens
    multiplier_spawn_interval = 3.0  # Seconds between spawns
    last_multiplier_spawn_time = time.time()
    collision_activation_time = 0.2    # Seconds required to activate a multiplier
    active_multiplier_value = 1.0      # Currently active multiplier factor (default 1.0)
    active_multiplier_timer = 0.0      # Remaining time (in seconds) for active multiplier
    active_multiplier_duration = 5.0   # When activated, multiplier lasts for 5 seconds
    token_expiry_duration = 7.0        # Tokens expire after 7 seconds if not collected

    # Power pose settings
    power_poses = define_power_poses(mp_holistic)
    active_power_pose = None
    power_pose_interval_min = 15.0  # Min seconds between power pose challenges
    power_pose_interval_max = 30.0  # Max seconds between power pose challenges
    next_power_pose_time = time.time() + random.uniform(power_pose_interval_min, power_pose_interval_max)
    pose_reference_images = create_pose_references()

    # For delta time calculation
    last_frame_time = time.time()
    frame_id = 0
    
    # Thread synchronization
    frame_lock = threading.Lock()
    results_lock = threading.Lock()
    multipliers_lock = threading.Lock()
    score_lock = threading.Lock()
    
    # Initialize parallel processors
    frame_processor = FrameProcessor(mp_holistic)
    motion_detector = MotionDetector(threshold_value=movement_threshold_value)
    effects_processor = EffectsProcessor()
    
    # Start processor threads
    frame_processor.start()
    motion_detector.start()
    effects_processor.start()
    
    # Number of CPU cores for thread pool
    num_cores = max(2, cpu_count() - 1)  # Leave one core free for UI
    print(f"Using {num_cores} cores for parallel processing")
    
    # Thread pool for general parallel tasks
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_cores)
    
    # For storing intermediate processing results
    current_results = None
    current_threshold = None
    current_threshold_colored = None
    current_frame_with_effects = None
    
    # Main game loop
    try:
        while True:
            current_time = time.time()
            dt = current_time - last_frame_time
            last_frame_time = current_time

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            frame = cv2.flip(frame, 1)
            
            # Increment frame ID and create copies for different pipelines
            frame_id += 1
            frame_copy = frame.copy()
            
            # Submit frame to parallel processing pipelines
            frame_processor.submit_frame(frame_id, frame.copy())
            motion_detector.submit_frame(frame_id, frame_copy)
            
            # Try to get latest results from MediaPipe processing
            mp_result = frame_processor.get_result(timeout=0.01)
            if mp_result is not None:
                result_frame_id, current_results = mp_result
            
            # Try to get latest motion detection results
            motion_result = motion_detector.get_result(timeout=0.01)
            if motion_result is not None:
                motion_frame_id, current_threshold, current_threshold_colored = motion_result
            
            # Initial variables for processing
            bbox = None
            landmarks = None
            movement = 0
            
            # Draw MediaPipe landmarks for visual feedback if we have results
            if current_results is not None:
                if current_results.face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=current_results.face_landmarks,
                        connections=mp_holistic.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                if current_results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=current_results.pose_landmarks,
                        connections=mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
                
                # Get normalized pose landmarks for game logic
                landmarks = []
                if current_results.pose_landmarks:
                    for landmark in current_results.pose_landmarks.landmark:
                        landmarks.append(landmark)
                    
                    # Calculate bounding box around person
                    x_coordinates = [landmark.x for landmark in landmarks]
                    y_coordinates = [landmark.y for landmark in landmarks]
                    
                    if x_coordinates and y_coordinates:
                        x_min, x_max = min(x_coordinates), max(x_coordinates)
                        y_min, y_max = min(y_coordinates), max(y_coordinates)
                        
                        # Convert normalized coordinates to pixel coordinates
                        h, w = frame.shape[:2]
                        bbox = {
                            'x_min': int(x_min * w),
                            'y_min': int(y_min * h),
                            'x_max': int(x_max * w),
                            'y_max': int(y_max * h),
                            'width': int((x_max - x_min) * w),
                            'height': int((y_max - y_min) * h),
                            'center_x': int((x_min + x_max) / 2 * w),
                            'center_y': int((y_min + y_max) / 2 * h)
                        }

            # Process movement detection
            if current_threshold is not None:
                # Calculate movement as the percentage of white pixels in the threshold image
                movement = np.sum(current_threshold == 255) / (current_threshold.shape[0] * current_threshold.shape[1])
                movement = movement * 100  # Convert to percentage
                
                # Display the motion detection overlay
                if current_threshold_colored is not None:
                    motion_overlay = current_threshold_colored.copy()
                    motion_overlay = cv2.addWeighted(frame, 0.7, motion_overlay, 0.3, 0)
                    
                    # In freeze phase, show the motion overlay more prominently
                    if music_state == "freeze":
                        frame = cv2.addWeighted(frame, 0.6, current_threshold_colored, 0.4, 0)
            
            # Game state updates
            with score_lock:
                # Update score based on movement (only in dance phase)
                if music_state == "dance":
                    score += movement * score_scaling * active_multiplier_value * dt
                
                # Update multiplier timer
                if active_multiplier_timer > 0:
                    active_multiplier_timer -= dt
                    if active_multiplier_timer <= 0:
                        active_multiplier_value = 1.0  # Reset multiplier when time expires
            
            # Check for music state change
            if current_time >= next_state_change_time:
                if music_state == "dance":
                    # Switch to freeze
                    music_state = "freeze"
                    pygame.mixer.music.pause()
                    freeze_duration = random.uniform(3, 6)  # Random freeze duration
                    next_state_change_time = current_time + freeze_duration
                    
                    # Add a "freeze" visual effect
                    effects_processor.add_effect(VisualEffect("flash", duration=0.5, intensity=1.5, start_time=current_time))
                    effects_processor.add_effect(VisualEffect("vignette", duration=freeze_duration, intensity=0.8, start_time=current_time))
                else:
                    # Switch to dance
                    music_state = "dance"
                    pygame.mixer.music.unpause()
                    dance_duration = random.uniform(15, 25)  # Random dance duration
                    next_state_change_time = current_time + dance_duration
                    
                    # Add a "resume" visual effect
                    effects_processor.add_effect(VisualEffect("flash", duration=0.3, intensity=1.0, start_time=current_time))
                    effects_processor.add_effect(VisualEffect("particles", duration=1.5, intensity=1.0, start_time=current_time))
            
            # Power pose challenge logic
            if active_power_pose is None and current_time >= next_power_pose_time:
                # Start a new power pose challenge
                active_power_pose = random.choice(power_poses)
                active_power_pose.reset(current_time)
                
                # Add visual effect for new pose challenge
                effects_processor.add_effect(VisualEffect("flash", duration=0.5, intensity=1.0, start_time=current_time))
                
            if active_power_pose is not None:
                # Check if pose challenge has expired
                if active_power_pose.is_expired(current_time):
                    active_power_pose = None
                    next_power_pose_time = current_time + random.uniform(power_pose_interval_min, power_pose_interval_max)
                else:
                    # Update pose progress
                    if landmarks:
                        pose_completed = active_power_pose.update(landmarks, dt)
                        
                        if pose_completed:
                            # Award bonus points for completing the pose
                            with score_lock:
                                score += active_power_pose.bonus_score
                            
                            # Add visual celebration effects
                            effects_processor.add_effect(VisualEffect("particles", duration=2.0, intensity=1.5, start_time=current_time))
                            effects_processor.add_effect(VisualEffect("flash", duration=0.3, intensity=0.7, start_time=current_time))
                            
                            # Reset for next pose challenge
                            active_power_pose = None
                            next_power_pose_time = current_time + random.uniform(power_pose_interval_min, power_pose_interval_max)
            
            # Multiplier token spawn and management
            if current_time - last_multiplier_spawn_time > multiplier_spawn_interval and music_state == "dance":
                # Spawn a new multiplier
                x = random.randint(50, 590)
                y = 50  # Start at top of screen
                value = random.choice([1.5, 2.0, 3.0])  # Different multiplier values
                
                with multipliers_lock:
                    multipliers.append(Multiplier(x, y, value, created_time=current_time))
                
                last_multiplier_spawn_time = current_time
                multiplier_spawn_interval = random.uniform(2.0, 5.0)  # Random interval for next spawn
            
            # Update all multipliers
            with multipliers_lock:
                for multiplier in multipliers[:]:  # Use slice copy for safe removal during iteration
                    multiplier.update()
                    
                    # Remove if off-screen or expired
                    if multiplier.y > 480 or (current_time - multiplier.created_time > token_expiry_duration):
                        multipliers.remove(multiplier)
                        continue
                    
                    # Check for collision with player
                    if bbox is not None:
                        player_x = bbox['center_x']
                        player_y = bbox['center_y']
                        
                        # Simple circle-point collision
                        distance = math.sqrt((player_x - multiplier.x)**2 + (player_y - multiplier.y)**2)
                        
                        if distance < multiplier.radius + bbox['width'] * 0.4:  # Adjust collision radius based on player width
                            # Start counting collision time
                            multiplier.collision_time += dt
                            
                            # Visual feedback for collision
                            cv2.circle(frame, (multiplier.x, multiplier.y), 
                                      int(multiplier.radius * (1 + multiplier.collision_time)), 
                                      (0, 255, 255), 2)
                            
                            # Activate multiplier after sufficient collision time
                            if multiplier.collision_time >= collision_activation_time:
                                with score_lock:
                                    active_multiplier_value = multiplier.value
                                    active_multiplier_timer = active_multiplier_duration
                                
                                # Add more noticeable effects
                                effects_processor.add_effect(VisualEffect("flash", duration=0.5, intensity=0.8, start_time=current_time))
                                effects_processor.add_effect(VisualEffect("particles", duration=1.0, intensity=1.2, start_time=current_time))
                                
                                # Print debug info to console
                                print(f"Multiplier activated: x{multiplier.value} for {active_multiplier_duration}s")
                                
                                multipliers.remove(multiplier)
                        else:
                            # Reset collision time if no longer colliding
                            multiplier.collision_time = 0.0
            
            # Draw multipliers on screen
            with multipliers_lock:
                for multiplier in multipliers:
                    # Draw multiplier token
                    cv2.circle(frame, (multiplier.x, multiplier.y), multiplier.radius + 3, (0, 255, 255), -1)
                    cv2.circle(frame, (multiplier.x, multiplier.y), multiplier.radius + 5, (0, 0, 0), 2)
                    
                    # Draw multiplier value
                    cv2.putText(frame, f"x{multiplier.value}", 
                               (multiplier.x - 15, multiplier.y + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Prepare frame for visual effects
            effects_processor.submit_frame(frame_id, frame, dt)
            
            # Try to get processed frame with effects
            effect_result = effects_processor.get_result(timeout=0.01)
            if effect_result is not None:
                effect_frame_id, current_frame_with_effects = effect_result
                frame = current_frame_with_effects
            
            # Draw UI elements
            
            # Draw game state banner
            if music_state == "dance":
                cv2.rectangle(frame, (0, 0), (640, 40), (0, 255, 0), -1)
                cv2.putText(frame, "DANCE! Move to score points!", (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            else:  # freeze state
                cv2.rectangle(frame, (0, 0), (640, 40), (0, 0, 255), -1)
                cv2.putText(frame, "FREEZE! Don't move!", (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # In freeze state, check if player is moving too much
                if movement > 5.0:  # Threshold for excessive movement
                    with score_lock:
                        penalty = movement * 0.5  # Penalty proportional to movement
                        score = max(0, score - penalty * dt)  # Apply penalty but don't go below 0
                    
                    # Visual feedback for penalty
                    cv2.putText(frame, "Penalty! Stay still!", (200, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # Draw score
            with score_lock:
                score_text = f"Score: {int(score)}"
                cv2.putText(frame, score_text, (20, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Debug: Multipliers: {len(multipliers)} Active: {active_multiplier_value:.1f}", 
                           (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
                
                # Draw active multiplier if any
                if active_multiplier_timer > 0:
                    multiplier_text = f"Multiplier: x{active_multiplier_value} ({int(active_multiplier_timer)}s)"
                    cv2.putText(frame, multiplier_text, (20, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Draw power pose challenge if active
            if active_power_pose is not None:
                # Draw challenge banner
                cv2.rectangle(frame, (0, 430), (640, 480), (255, 165, 0), -1)
                cv2.putText(frame, f"POWER POSE: {active_power_pose.name}", (20, 450), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(frame, active_power_pose.description, (20, 470), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Draw progress bar
                progress_width = int(200 * active_power_pose.progress)
                cv2.rectangle(frame, (400, 445), (400 + progress_width, 465), (0, 255, 0), -1)
                cv2.rectangle(frame, (400, 445), (600, 465), (255, 255, 255), 2)
                
                # Draw pose reference image
                if active_power_pose.name in pose_reference_images:
                    ref_img = pose_reference_images[active_power_pose.name]
                    # Resize reference image to a small thumbnail
                    ref_img = cv2.resize(ref_img, (100, 100))
                    
                    # Position in top-right corner
                    frame[50:150, 520:620] = ref_img
            
            # Display movement level meter (only in dance phase)
            if music_state == "dance":
                meter_width = int(movement * 5)  # Scale for better visibility
                meter_width = min(200, meter_width)  # Cap at 200 pixels
                cv2.rectangle(frame, (400, 70), (400 + meter_width, 90), (0, 255, 0), -1)
                cv2.rectangle(frame, (400, 70), (600, 90), (255, 255, 255), 2)
                cv2.putText(frame, "Movement", (320, 85), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the resulting frame
            cv2.imshow('Dance Freeze Game', frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error in main game loop: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up resources
        print("Cleaning up resources...")
        
        # Stop processing threads
        frame_processor.stop()
        motion_detector.stop()
        effects_processor.stop()
        
        # Shutdown thread pool
        executor.shutdown()
        
        # Release OpenCV resources
        cap.release()
        cv2.destroyAllWindows()
        
        # Stop music
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        
        print(f"Game Over! Final Score: {int(score)}")

if __name__ == "__main__":
    main()