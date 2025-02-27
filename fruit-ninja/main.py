import cv2
import mediapipe as mp
import pygame
import random
import math

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

    # Project AC onto AB, computing parameter t along AB.
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
        # For image assets, use:
        # screen.blit(self.image, (int(self.x - self.radius), int(self.y - self.radius)))
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

class Bomb(Fruit):
    def __init__(self, x, y, radius, color, vx, vy):
        super().__init__(x, y, radius, color, vx, vy)
        self.is_bomb = True
        # For bomb image assets, load here:
        # self.image = pygame.image.load("path/to/bomb.png")
        # self.image = pygame.transform.scale(self.image, (2*self.radius, 2*self.radius))

    def draw(self, screen):
        # For image assets, use:
        # screen.blit(self.image, (int(self.x - self.radius), int(self.y - self.radius)))
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
    hands = mp_hands.Hands(min_detection_confidence=0.7,
                           min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(10)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set webcam resolution to match the Pygame window.
    screen_width, screen_height = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Fruit Ninja with Hand Tracking")
    clock = pygame.time.Clock()

    score = 0
    fruits = []
    spawn_timer = 0.0
    slice_threshold = 40  # Minimum movement (pixels) to register a slice.
    prev_smoothed_tip = None  # Previous smoothed index finger tip.
    slicing_line = None
    game_time = 0.0  # Total game time.

    # Smoothing factor for the finger tracking (0 < alpha <= 1)
    alpha = 0.3
    smoothed_tip = None

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
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Get raw index finger tip position.
        raw_tip = None
        if results.multi_hand_landmarks:
            handLms = results.multi_hand_landmarks[0]
            h, w, _ = frame.shape
            index_tip = handLms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            raw_tip = (int(index_tip.x * w), int(index_tip.y * h))

        # Apply exponential smoothing to reduce zigzag jitter.
        if raw_tip:
            if smoothed_tip is None:
                smoothed_tip = raw_tip
            else:
                smoothed_tip = (
                    int(alpha * raw_tip[0] + (1 - alpha) * smoothed_tip[0]),
                    int(alpha * raw_tip[1] + (1 - alpha) * smoothed_tip[1])
                )
        else:
            smoothed_tip = None

        # Determine slicing gesture using smoothed coordinates.
        slicing_line = None
        if prev_smoothed_tip and smoothed_tip:
            dx = smoothed_tip[0] - prev_smoothed_tip[0]
            dy = smoothed_tip[1] - prev_smoothed_tip[1]
            distance = math.hypot(dx, dy)
            if distance > slice_threshold:
                slicing_line = (prev_smoothed_tip, smoothed_tip)
        prev_smoothed_tip = smoothed_tip

        # Update fruits and bombs.
        for obj in fruits:
            obj.update(dt)
        fruits = [obj for obj in fruits if obj.y - obj.radius < screen_height]

        # Check for collision with slicing line.
        if slicing_line:
            A, B = slicing_line
            for obj in fruits:
                if not obj.sliced and line_circle_collision(A, B, (obj.x, obj.y), obj.radius):
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

        screen.fill((0, 0, 0))
        for obj in fruits:
            obj.draw(screen)

        if slicing_line:
            pygame.draw.line(screen, (0, 255, 0), slicing_line[0], slicing_line[1], 5)

        # Draw the (smoothed) index finger tip.
        if smoothed_tip:
            pygame.draw.circle(screen, (255, 0, 0), smoothed_tip, 10)

        font = pygame.font.SysFont(None, 36)
        score_text = font.render("Score: " + str(score), True, (255, 255, 255))
        screen.blit(score_text, (10, 10))

        pygame.display.flip()

    cap.release()
    pygame.quit()

if __name__ == '__main__':
    main()
