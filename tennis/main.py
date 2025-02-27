import cv2
import mediapipe as mp
import pygame
import sys
import math
import random

# ------------------------
# Configuration & Constants
# ------------------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
SCREEN_SIZE = (FRAME_WIDTH, FRAME_HEIGHT)

# Colors (RGB)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED   = (255, 0, 0)
BLUE  = (0, 0, 255)

# Racket settings
RACKET_WIDTH = 50
RACKET_HEIGHT = 10
SWING_THRESHOLD = 20  # pixel movement threshold to detect a swing

# Ball settings
BALL_RADIUS = 10
BALL_SPEED_X = 5
BALL_SPEED_Y = 5

# ------------------------
# Initialize Pygame
# ------------------------
pygame.init()
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption("MediaPipe Tennis Game")
clock = pygame.time.Clock()

# ------------------------
# Initialize MediaPipe and OpenCV
# ------------------------
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose()
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(10)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# ------------------------
# Game Object Initializations
# ------------------------
# Use the nose landmark for the player's body position.
player_pos = [SCREEN_SIZE[0] // 2, SCREEN_SIZE[1] // 2]

# Racket position (will be updated using hand landmark)
racket_pos = [0, 0]

# Ball position and velocity
ball_pos = [SCREEN_SIZE[0] // 2, SCREEN_SIZE[1] // 2]
ball_vx = BALL_SPEED_X
ball_vy = BALL_SPEED_Y

# Hand tracking for swing detection
last_hand_pos = None
swing_detected = False

# ------------------------
# Main Game Loop
# ------------------------
running = True
while running:
    # Handle Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)  # Mirror image for natural movement
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe
    pose_results = pose.process(image_rgb)
    hands_results = hands.process(image_rgb)

    # ------------------------
    # Update Player Position Using Pose Landmarks
    # ------------------------
    if pose_results.pose_landmarks:
        # Using the nose as a reference for the player's position
        nose = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        player_pos[0] = int(nose.x * FRAME_WIDTH)
        player_pos[1] = int(nose.y * FRAME_HEIGHT)

    # ------------------------
    # Update Racket Position and Detect Swing Using Hand Landmarks
    # ------------------------
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            # Use index finger tip (landmark index 8) for the racket position
            hand_x = int(hand_landmarks.landmark[8].x * FRAME_WIDTH)
            hand_y = int(hand_landmarks.landmark[8].y * FRAME_HEIGHT)
            racket_pos[0] = hand_x - RACKET_WIDTH // 2
            racket_pos[1] = hand_y - RACKET_HEIGHT // 2

            # Compute movement speed for swing detection
            if last_hand_pos is not None:
                dx = hand_x - last_hand_pos[0]
                dy = hand_y - last_hand_pos[1]
                speed = math.sqrt(dx * dx + dy * dy)
                if speed > SWING_THRESHOLD:
                    swing_detected = True
                else:
                    swing_detected = False
            last_hand_pos = (hand_x, hand_y)
            break  # Use only the first detected hand
    else:
        swing_detected = False

    # ------------------------
    # Update Ball Position & Check Collisions
    # ------------------------
    # Move ball
    ball_pos[0] += ball_vx
    ball_pos[1] += ball_vy

    # Bounce off screen boundaries
    if ball_pos[0] <= BALL_RADIUS or ball_pos[0] >= SCREEN_SIZE[0] - BALL_RADIUS:
        ball_vx = -ball_vx
    if ball_pos[1] <= BALL_RADIUS or ball_pos[1] >= SCREEN_SIZE[1] - BALL_RADIUS:
        ball_vy = -ball_vy

    # Create rectangles for collision detection
    racket_rect = pygame.Rect(racket_pos[0], racket_pos[1], RACKET_WIDTH, RACKET_HEIGHT)
    ball_rect = pygame.Rect(ball_pos[0] - BALL_RADIUS, ball_pos[1] - BALL_RADIUS,
                            BALL_RADIUS * 2, BALL_RADIUS * 2)
    if racket_rect.colliderect(ball_rect):
        # If a swing is detected when the ball hits the racket, change the ball's velocity more aggressively.
        if swing_detected:
            ball_vx = random.choice([-1, 1]) * abs(ball_vx)
            ball_vy = -abs(ball_vy)  # Send ball upward
        else:
            ball_vy = -ball_vy  # Simple bounce if no swing

    # ------------------------
    # Render Game Objects with Pygame
    # ------------------------
    screen.fill(BLACK)

    # Draw the player's body (a circle)
    pygame.draw.circle(screen, GREEN, (player_pos[0], player_pos[1]), 15)

    # Draw the racket (a rectangle)
    pygame.draw.rect(screen, BLUE, racket_rect)

    # Draw the ball
    pygame.draw.circle(screen, RED, (int(ball_pos[0]), int(ball_pos[1])), BALL_RADIUS)

    # Optional: Display "Swing!" text when a swing is detected
    if swing_detected:
        font = pygame.font.SysFont(None, 36)
        text = font.render("Swing!", True, WHITE)
        screen.blit(text, (10, 10))

    pygame.display.flip()
    clock.tick(30)  # Limit the game to 30 FPS

# ------------------------
# Clean Up
# ------------------------
cap.release()
pygame.quit()
sys.exit()
