import cv2
import mediapipe as mp
import pygame
import random
import math
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

# -------------------------------
# Helper Functions and Classes
# -------------------------------

def line_circle_collision(A, B, circle_center, radius):
    """
    Check if the line segment AB intersects a circle defined by circle_center and radius.
    Even if the line just touches the circle boundary, it's considered a collision.
    """
    (x1, y1) = A
    (x2, y2) = B
    (cx, cy) = circle_center

    # Vector from A to B.
    ABx = x2 - x1
    ABy = y2 - y1

    # Vector from A to circle center.
    ACx = cx - x1
    ACy = cy - y1

    ab2 = ABx**2 + ABy**2
    if ab2 == 0:
        return False  # Avoid division by zero.
    t = (ACx * ABx + ACy * ABy) / ab2
    t = max(0, min(1, t))  # Clamp t to [0, 1].

    # Closest point on AB to the circle center.
    closest_x = x1 + t * ABx
    closest_y = y1 + t * ABy

    # Distance from the circle center to this closest point.
    dist = math.sqrt((closest_x - cx) ** 2 + (closest_y - cy) ** 2)
    return dist <= radius

class Fruit:
    def __init__(self, x, y, radius, color, vx, vy):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.vx = vx  # Horizontal velocity (pixels/second)
        self.vy = vy  # Vertical velocity (pixels/second)
        self.sliced = False
        self.is_bomb = False
        # For image assets, load here:
        # self.image = pygame.image.load("path/to/fruit.png")
        # self.image = pygame.transform.scale(self.image, (2*self.radius, 2*self.radius))

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        gravity = 500  # pixels per second^2
        self.vy += gravity * dt

    def draw(self, screen):
        # For image assets, use screen.blit() instead.
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

class Bomb(Fruit):
    def __init__(self, x, y, radius, color, vx, vy):
        super().__init__(x, y, radius, color, vx, vy)
        self.is_bomb = True
        # For bomb image assets, load here:
        # self.image = pygame.image.load("path/to/bomb.png")
        # self.image = pygame.transform.scale(self.image, (2*self.radius, 2*self.radius))

    def draw(self, screen):
        # For image assets, use screen.blit() instead.
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(screen, (255, 0, 0), (int(self.x), int(self.y)), self.radius, 2)

def spawn_fruit(screen_width, screen_height, multiplier):
    x = random.randint(50, screen_width - 50)
    y = screen_height + 50
    radius = random.randint(20, 30)
    color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
    vx = random.uniform(-100, 100) * multiplier
    vy = random.uniform(-600, -400) * multiplier
    return Fruit(x, y, radius, color, vx, vy)

def spawn_bomb(screen_width, screen_height, multiplier):
    x = random.randint(50, screen_width - 50)
    y = screen_height + 50
    radius = random.randint(20, 30)
    color = (0, 0, 0)  # Bomb color.
    vx = random.uniform(-100, 100) * multiplier
    vy = random.uniform(-600, -400) * multiplier
    return Bomb(x, y, radius, color, vx, vy)

# -------------------------------
# Main Game Function
# -------------------------------

def main():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.7,
                           min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)  # Change to 0 if needed.
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set webcam resolution to match the Pygame window.
    screen_width, screen_height = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Fruit Ninja with Whole Hand Tracking")
    clock = pygame.time.Clock()

    score = 0
    fruits = []
    spawn_timer = 0.0
    game_time = 0.0  # Total game time.

    # Smoothing factor for hand centroid (0 < alpha <= 1)
    alpha = 0.3

    # Variables for hand centroid and in-game pointer.
    hand_centroid = None      # Smoothed hand centroid (from raw landmarks).
    prev_hand_centroid = None # Previous smoothed hand centroid.
    game_pointer = None       # In-game pointer that moves faster.
    scaling_factor = 1.5      # 1x hand movement produces 1.5x in-game movement.

    # Trail for swipe: maintain a list of recent game_pointer positions.
    trail_points = []
    max_trail_length = 5  # Adjust for a longer/shorter trail.

    # Lower swipe threshold for increased sensitivity.
    slice_threshold = 15

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        game_time += dt

        # Slowly increase speed multiplier (capped at 2.0).
        multiplier = 1 + (game_time / 120.0)
        multiplier = min(multiplier, 2.0)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        ret, frame = cap.read()
        if not ret:
            continue

        # Flip and convert frame.
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Draw landmarks on the frame for a separate "Hand Detection" window.
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
        cv2.imshow("Hand Detection", frame)
        cv2.waitKey(1)

        # Compute the centroid of the whole hand.
        raw_pointer = None
        if results.multi_hand_landmarks:
            handLms = results.multi_hand_landmarks[0]
            h, w, _ = frame.shape
            sum_x, sum_y, count = 0, 0, 0
            for lm in handLms.landmark:
                sum_x += lm.x
                sum_y += lm.y
                count += 1
            avg_x = int((sum_x / count) * w)
            avg_y = int((sum_y / count) * h)
            raw_pointer = (avg_x, avg_y)

        # Apply exponential smoothing to the raw pointer.
        if raw_pointer:
            if hand_centroid is None:
                hand_centroid = raw_pointer
            else:
                hand_centroid = (
                    int(alpha * raw_pointer[0] + (1 - alpha) * hand_centroid[0]),
                    int(alpha * raw_pointer[1] + (1 - alpha) * hand_centroid[1])
                )
        else:
            hand_centroid = None

        # Update the in-game pointer using the change in the hand centroid.
        if hand_centroid:
            if game_pointer is None:
                game_pointer = hand_centroid
            else:
                if prev_hand_centroid:
                    # Compute displacement from the hand centroid change.
                    dx = hand_centroid[0] - prev_hand_centroid[0]
                    dy = hand_centroid[1] - prev_hand_centroid[1]
                    # Scale the displacement.
                    scaled_dx = dx * scaling_factor
                    scaled_dy = dy * scaling_factor
                    game_pointer = (int(game_pointer[0] + scaled_dx), int(game_pointer[1] + scaled_dy))
            prev_hand_centroid = hand_centroid
        else:
            game_pointer = None

        # Add current game_pointer to the trail.
        if game_pointer:
            trail_points.append(game_pointer)
            if len(trail_points) > max_trail_length:
                trail_points.pop(0)

        # Determine the most recent swipe segment.
        slicing_line = None
        if len(trail_points) >= 2:
            slicing_line = (trail_points[-2], trail_points[-1])

        # Update fruits and bombs.
        for obj in fruits:
            obj.update(dt)
        fruits = [obj for obj in fruits if obj.y - obj.radius < screen_height]

        # Check for collision using the swipe segment.
        if slicing_line:
            A, B = slicing_line
            for obj in fruits:
                if not obj.sliced and line_circle_collision(A, B, (obj.x, obj.y), obj.radius):
                    obj.sliced = True
                    if obj.is_bomb:
                        score -= 1
                    else:
                        score += 1

        # Additionally, if the in-game pointer directly touches any fruit.
        if game_pointer:
            for obj in fruits:
                if not obj.sliced:
                    dist = math.hypot(game_pointer[0] - obj.x, game_pointer[1] - obj.y)
                    if dist <= obj.radius:
                        obj.sliced = True
                        if obj.is_bomb:
                            score -= 1
                        else:
                            score += 1

        fruits = [obj for obj in fruits if not obj.sliced]

        spawn_timer += dt
        if spawn_timer > 1.0:
            spawn_timer = 0.0
            if random.random() < 0.8:
                fruits.append(spawn_fruit(screen_width, screen_height, multiplier))
            else:
                fruits.append(spawn_bomb(screen_width, screen_height, multiplier))

        # Draw game elements.
        screen.fill((0, 0, 0))
        for obj in fruits:
            obj.draw(screen)

        # Draw the swipe trail (using anti-aliased lines for smoothness).
        if len(trail_points) > 1:
            pygame.draw.aalines(screen, (0, 255, 0), False, trail_points)

        # Optionally, also draw the most recent segment thicker.
        if slicing_line:
            pygame.draw.line(screen, (0, 255, 0), slicing_line[0], slicing_line[1], 3)

        # Draw the in-game pointer.
        if game_pointer:
            pygame.draw.circle(screen, (255, 0, 0), game_pointer, 10)

        font = pygame.font.SysFont(None, 36)
        score_text = font.render("Score: " + str(score), True, (255, 255, 255))
        screen.blit(score_text, (10, 10))

        pygame.display.flip()

    cap.release()
    pygame.quit()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
