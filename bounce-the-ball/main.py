import cv2
import mediapipe as mp
import pygame
import threading
import time
import random
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

# ------------------------------
# Module 1: Hand Tracking with MediaPipe
# ------------------------------

class HandTracker(threading.Thread):
    """
    This class uses MediaPipe to track the player's hand from the webcam.
    It continuously updates the y-coordinate of the index finger tip (landmark 8),
    which we use to control the player's racket in the game.
    """
    def __init__(self, display=False):
        super(HandTracker, self).__init__()
        self.display = display
        self.cap = cv2.VideoCapture(10)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

        self.hand_y = None
        self.running = True
        self.lock = threading.Lock()

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    h, w, _ = frame.shape
                    index_finger_tip = hand_landmarks.landmark[8]
                    cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                    with self.lock:
                        self.hand_y = cy

                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            if self.display:
                cv2.imshow("Hand Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False

        self.cap.release()
        cv2.destroyAllWindows()

    def get_hand_y(self):
        with self.lock:
            return self.hand_y

    def stop(self):
        self.running = False

# ------------------------------
# Module 2: Tennis Game with Pygame and Improved AI
# ------------------------------

class TennisGame:
    """
    A simple tennis game built with Pygame.
    The player’s racket is controlled by the HandTracker.
    The opponent’s racket automatically follows the ball with improved AI.
    """
    def __init__(self, hand_tracker):
        self.hand_tracker = hand_tracker
        pygame.init()

        self.WIDTH, self.HEIGHT = 800, 600
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Tennis Game with Hand Tracking")
        self.clock = pygame.time.Clock()

        self.ball = pygame.Rect(self.WIDTH // 2 - 10, self.HEIGHT // 2 - 10, 20, 20)
        self.ball_speed_x = 4
        self.ball_speed_y = 4

        self.player_racket = pygame.Rect(30, self.HEIGHT // 2 - 50, 10, 100)
        self.opponent_racket = pygame.Rect(self.WIDTH - 40, self.HEIGHT // 2 - 50, 10, 100)

        self.player_score = 0
        self.opponent_score = 0
        self.font = pygame.font.SysFont("Arial", 30)

    def reset_ball(self):
        """Reset the ball to the center and reverse its x-direction."""
        self.ball.center = (self.WIDTH // 2, self.HEIGHT // 2)
        self.ball_speed_x = 4 * (-1 if random.random() > 0.5 else 1)
        self.ball_speed_y = 4 * (-1 if random.random() > 0.5 else 1)

    def draw(self):
        """Draw game objects and score on the screen."""
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 255, 255), self.player_racket)
        pygame.draw.rect(self.screen, (255, 0, 0), self.opponent_racket)
        pygame.draw.ellipse(self.screen, (0, 255, 255), self.ball)
        pygame.draw.aaline(self.screen, (255, 255, 255), (self.WIDTH // 2, 0), (self.WIDTH // 2, self.HEIGHT))

        player_text = self.font.render(f"{self.player_score}", False, (255, 255, 255))
        opponent_text = self.font.render(f"{self.opponent_score}", False, (255, 255, 255))
        self.screen.blit(player_text, (self.WIDTH // 4, 20))
        self.screen.blit(opponent_text, (self.WIDTH * 3 // 4, 20))

        pygame.display.flip()

    def move_ball(self):
        """Update the ball position and increase speed after racket bounces."""
        self.ball.x += self.ball_speed_x
        self.ball.y += self.ball_speed_y

        if self.ball.top <= 0 or self.ball.bottom >= self.HEIGHT:
            self.ball_speed_y *= -1

        if self.ball.colliderect(self.player_racket) or self.ball.colliderect(self.opponent_racket):
            self.ball_speed_x *= -1

            speed_increment = 0.3  
            max_speed = 10
            self.ball_speed_x = max(min(self.ball_speed_x + speed_increment * (1 if self.ball_speed_x > 0 else -1), max_speed), -max_speed)
            self.ball_speed_y = max(min(self.ball_speed_y + speed_increment * (1 if self.ball_speed_y > 0 else -1), max_speed), -max_speed)

        if self.ball.left <= 0:
            self.opponent_score += 1
            self.reset_ball()
        if self.ball.right >= self.WIDTH:
            self.player_score += 1
            self.reset_ball()

    def move_opponent(self):
        """Improved AI that reacts 90% of the time and moves faster towards the ball."""
        reaction_chance = 0.9  

        if random.random() < reaction_chance:
            if self.opponent_racket.centery < self.ball.centery:
                self.opponent_racket.y += min(abs(self.ball_speed_y), 6)
            elif self.opponent_racket.centery > self.ball.centery:
                self.opponent_racket.y -= min(abs(self.ball_speed_y), 6)

        self.opponent_racket.y = max(0, min(self.HEIGHT - self.opponent_racket.height, self.opponent_racket.y))

    def update_player_racket(self):
        """Update player's racket based on hand position."""
        hand_y = self.hand_tracker.get_hand_y()
        if hand_y is not None:
            self.player_racket.centery = hand_y
            if self.player_racket.top < 0:
                self.player_racket.top = 0
            if self.player_racket.bottom > self.HEIGHT:
                self.player_racket.bottom = self.HEIGHT

    def run(self):
        """Main game loop."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.update_player_racket()
            self.move_ball()
            self.move_opponent()
            self.draw()
            self.clock.tick(60)

        pygame.quit()

# ------------------------------
# Main Integration
# ------------------------------

def main():
    hand_tracker = HandTracker(display=True)  
    hand_tracker.start()

    game = TennisGame(hand_tracker)
    try:
        game.run()
    finally:
        hand_tracker.stop()
        hand_tracker.join()

if __name__ == '__main__':
    main()
