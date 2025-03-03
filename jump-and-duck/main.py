import cv2
import mediapipe as mp
import pygame
import sys
import random
import threading

# Import our sprite classes from sprites.py
from sprites import SpriteSheet, Dino, Obstacle, Crow

# Global action variable used by the game loop.
action = "none"  # "jump", "duck", or "none"

# ---------------------------
# Pose detection using OpenCV & MediaPipe
# ---------------------------
def pose_detection():
    global action
    cap = cv2.VideoCapture(0)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    baseline_set = False
    baseline_sum = 0
    frame_count = 0
    baseline_y = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame for a mirror view and convert color
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            # Using the NOSE landmark as a proxy for head position.
            nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            h, w, _ = frame.shape
            nose_y = nose.y * h

            if not baseline_set:
                baseline_sum += nose_y
                frame_count += 1
                if frame_count >= 30:  # use first 30 frames to compute baseline
                    baseline_y = baseline_sum / frame_count
                    baseline_set = True
                    print("Baseline head position (y):", baseline_y)
            else:
                # Determine action based on deviation from the baseline.
                if baseline_y - nose_y > 30:  # Head raised => jump
                    action = "jump"
                elif nose_y - baseline_y > 30:  # Head lowered => duck
                    action = "duck"
                else:
                    action = "none"

        cv2.imshow("Webcam (Press Q to exit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------------------
# Screen settings
# ---------------------------
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
GROUND_Y = 300

def main():
    global action
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Jump & Duck Game")
    clock = pygame.time.Clock()

    # Load the sprite sheet.
    # Change "assets_spritesheet.png" to your actual sprite sheet filename.
    sheet = SpriteSheet("200-offline-sprite.png")

    # Create the sprite-based Dino.
    dino = Dino(sheet)

    obstacles = []
    spawn_timer = 0
    score = 0
    game_over = False
    base_speed = 5

    # Main game loop.
    while True:
        clock.tick(30)  # 30 FPS

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if not game_over:
            # Increase obstacle speed gradually.
            current_speed = base_speed + (score // 300)

            # Pose-controlled actions.
            if action == "jump" and not dino.is_jumping:
                dino.is_jumping = True
                dino.velocity_y = dino.jump_speed
            dino.is_ducking = (action == "duck")

            # Update the dino's state (animation and physics).
            dino.update()

            # Spawn new obstacles at regular intervals.
            spawn_timer += 1
            if spawn_timer > 60:
                # 30% chance to spawn a crow (flying obstacle) vs ground obstacle.
                if random.random() < 0.3:
                    obstacles.append(Crow(sheet, SCREEN_WIDTH, current_speed))
                else:
                    obstacles.append(Obstacle(sheet, SCREEN_WIDTH, current_speed))
                spawn_timer = 0

            # Update obstacles and check for collisions.
            for obs in obstacles:
                obs.speed = current_speed  # update speed based on score
                obs.update()
                if obs.collides_with(dino):
                    game_over = True
            # Remove obstacles that have moved off screen.
            obstacles = [obs for obs in obstacles if not obs.off_screen()]

            score += 1

        # Drawing section.
        screen.fill((255, 255, 255))  # white background
        dino.draw(screen)
        for obs in obstacles:
            obs.draw(screen)
        # Draw the ground line.
        pygame.draw.line(screen, (0, 0, 0), (0, GROUND_Y), (SCREEN_WIDTH, GROUND_Y), 2)

        # Display the score.
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

if __name__ == "__main__":
    # Start pose detection in a separate thread (daemonized so it exits with the main program)
    cv_thread = threading.Thread(target=pose_detection, daemon=True)
    cv_thread.start()
    main()
