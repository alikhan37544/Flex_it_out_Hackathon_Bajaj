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
# Pose detection using OpenCV & MediaPipe (tracking head and upper body with dynamic baseline)
# ---------------------------
def pose_detection():
    global action
    cap = cv2.VideoCapture(0)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Use only upper-body landmarks.
    key_landmarks = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_EYE,
        mp_pose.PoseLandmark.RIGHT_EYE,
        mp_pose.PoseLandmark.LEFT_EAR,
        mp_pose.PoseLandmark.RIGHT_EAR,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
    ]

    # Thresholds (in pixels)
    jump_threshold = 20   # if average is at least 20px above baseline -> jump
    duck_threshold = 20   # if average is at least 20px below baseline -> duck
    alpha = 0.01          # smoothing factor for dynamic baseline update

    baseline = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame for mirror view and convert to RGB.
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        h, w, _ = frame.shape

        # Initialize baseline to the vertical midpoint if not set.
        if baseline is None:
            baseline = h / 2

        if results.pose_landmarks:
            total_y = 0
            count = 0
            for landmark in key_landmarks:
                lm = results.pose_landmarks.landmark[landmark]
                total_y += lm.y * h
                count += 1
            current_avg = total_y / count

            # Calculate difference between the current average and the baseline.
            diff = baseline - current_avg  # Positive: body raised (jump)

            # Always update baseline to adapt over time.
            baseline = (1 - alpha) * baseline + alpha * current_avg

            # Determine action based on the difference.
            if diff > jump_threshold:
                action = "jump"
            elif diff < -duck_threshold:
                action = "duck"
            else:
                action = "none"


        # Draw the dynamic baseline line and its value on the camera feed.
        cv2.line(frame, (0, int(baseline)), (w, int(baseline)), (0, 255, 0), 2)
        cv2.putText(frame, f"Baseline: {int(baseline)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Webcam (Press Q to exit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------------------
# Game classes using Pygame
# ---------------------------
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
        self.jump_speed = -15
        self.gravity = 1

    def update(self):
        if self.is_jumping:
            self.velocity_y += self.gravity
            self.y += self.velocity_y
            if self.y >= GROUND_Y:
                self.y = GROUND_Y
                self.is_jumping = False
                self.velocity_y = 0
        self.height = 30 if (self.is_ducking and not self.is_jumping) else 50

    def draw(self, screen):
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
        self.width = 30
        self.height = 20
        self.bottom = 260
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

    while True:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if not game_over:
            current_speed = base_speed + (score // 300)
            if action == "jump" and not dino.is_jumping:
                dino.is_jumping = True
                dino.velocity_y = dino.jump_speed
            dino.is_ducking = (action == "duck")
            dino.update()

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
    cv_thread = threading.Thread(target=pose_detection, daemon=True)
    cv_thread.start()
    main()
