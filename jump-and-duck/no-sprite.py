import cv2
import mediapipe as mp
import pygame
import sys
import random
import threading
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Global action variable used by the game loop.
action = "none"  # "jump", "duck", or "none"

# ---------------------------
# Pose detection using OpenCV & MediaPipe (dynamic change in y)
# ---------------------------
def pose_detection():
    global action
    cap = cv2.VideoCapture(1)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    prev_nose_y = None
    jump_threshold = 20   # If nose goes up by more than this many pixels (compared to previous frame), trigger jump
    duck_threshold = 20   # If nose goes down by more than this many pixels, trigger duck

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame for mirror view and convert color.
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            # Use the NOSE landmark as a proxy for head position.
            nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            h, w, _ = frame.shape
            nose_y = nose.y * h

            if prev_nose_y is None:
                prev_nose_y = nose_y
            else:
                diff = prev_nose_y - nose_y  # Positive diff: head is higher than previous frame.
                if diff > jump_threshold:
                    action = "jump"
                elif diff < -duck_threshold:
                    action = "duck"
                else:
                    action = "none"
                prev_nose_y = nose_y

        cv2.imshow("Webcam (Press Q to exit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------------------
# Game classes using Pygame
# ---------------------------
# Screen settings
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
GROUND_Y = 300

class Dino:
    def __init__(self):
        self.x = 50
        self.y = GROUND_Y
        self.width = 40
        self.height = 50
        self.velocity_y = 0
        self.is_jumping = False
        self.is_ducking = False
        self.jump_speed = -15  # initial upward speed
        self.gravity = 1

    def update(self):
        # Jumping physics
        if self.is_jumping:
            self.velocity_y += self.gravity
            self.y += self.velocity_y
            if self.y >= GROUND_Y:
                self.y = GROUND_Y
                self.is_jumping = False
                self.velocity_y = 0

        # Change hitbox if ducking (only when not jumping)
        if self.is_ducking and not self.is_jumping:
            self.height = 30
        else:
            self.height = 50

    def draw(self, screen):
        # Draw a simple green rectangle
        color = (0, 128, 0)
        pygame.draw.rect(screen, color, (self.x, self.y - self.height, self.width, self.height))

class Obstacle:
    def __init__(self, x, speed):
        self.x = x
        self.y = GROUND_Y
        self.width = 20
        self.height = random.randint(30, 50)
        self.speed = speed

    def update(self):
        self.x -= self.speed

    def draw(self, screen):
        # Draw obstacle as a red rectangle
        color = (128, 0, 0)
        pygame.draw.rect(screen, color, (self.x, self.y - self.height, self.width, self.height))

    def off_screen(self):
        return self.x + self.width < 0

    def collides_with(self, dino):
        dino_rect = pygame.Rect(dino.x, dino.y - dino.height, dino.width, dino.height)
        obs_rect = pygame.Rect(self.x, self.y - self.height, self.width, self.height)
        return dino_rect.colliderect(obs_rect)

class Crow:
    def __init__(self, x, speed):
        self.x = x
        # Position the crow to force a duck action.
        self.width = 30
        self.height = 20
        self.bottom = 260  # bottom position of the crow
        self.speed = speed

    def update(self):
        self.x -= self.speed

    def draw(self, screen):
        color = (0, 0, 0)
        pygame.draw.rect(screen, color, (self.x, self.bottom - self.height, self.width, self.height))

    def off_screen(self):
        return self.x + self.width < 0

    def collides_with(self, dino):
        dino_rect = pygame.Rect(dino.x, dino.y - dino.height, dino.width, dino.height)
        crow_rect = pygame.Rect(self.x, self.bottom - self.height, self.width, self.height)
        return dino_rect.colliderect(crow_rect)

# ---------------------------
# Main game loop using Pygame
# ---------------------------
def main():
    global action
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Jump & Duck Game")
    clock = pygame.time.Clock()

    dino = Dino()
    obstacles = []
    spawn_timer = 0
    score = 0
    game_over = False

    base_speed = 5

    # Game loop
    while True:
        clock.tick(30)  # 30 FPS

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if not game_over:
            current_speed = base_speed + (score // 300)

            # Trigger jump if action is "jump" and dino is not already jumping.
            if action == "jump" and not dino.is_jumping:
                dino.is_jumping = True
                dino.velocity_y = dino.jump_speed

            # Set ducking state if action is "duck".
            dino.is_ducking = (action == "duck")

            dino.update()

            # Spawn obstacles at intervals.
            spawn_timer += 1
            if spawn_timer > 60:
                if random.random() < 0.3:
                    obstacles.append(Crow(SCREEN_WIDTH, current_speed))
                else:
                    obstacles.append(Obstacle(SCREEN_WIDTH, current_speed))
                spawn_timer = 0

            for obs in obstacles:
                obs.speed = current_speed
                obs.update()
                if obs.collides_with(dino):
                    game_over = True
            obstacles = [obs for obs in obstacles if not obs.off_screen()]

            score += 1

        screen.fill((255, 255, 255))
        dino.draw(screen)
        for obs in obstacles:
            obs.draw(screen)
        pygame.draw.line(screen, (0, 0, 0), (0, GROUND_Y), (SCREEN_WIDTH, GROUND_Y), 2)

        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f"Score: {score}", True, (0, 0, 0))
        screen.blit(score_text, (600, 20))
        if game_over:
            over_text = font.render("Game Over! Press Q to Quit", True, (255, 0, 0))
            screen.blit(over_text, (200, 200))

        pygame.display.flip()

        if game_over:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]:
                pygame.quit()
                sys.exit()

# ---------------------------
# Start the pose detection thread and launch the game
# ---------------------------
if __name__ == "__main__":
    # Start pose detection in a separate thread.
    cv_thread = threading.Thread(target=pose_detection, daemon=True)
    cv_thread.start()
    main()
