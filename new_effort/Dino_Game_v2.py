import cv2
import mediapipe as mp
import pygame
import sys
import random
import threading
import os
from pygame import mixer

os.environ["QT_QPA_PLATFORM"] = "xcb"

# Global action variable used by the game loop.
action = "none"  # "jump", "duck", or "none"

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
GROUND_Y = 300
FPS = 60
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

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
        cv2.putText(frame, f"Action: {action}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Dino Game Controls (Press Q to exit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------------------
# Game classes using Pygame
# ---------------------------
class Dino:
    def __init__(self):
        self.x = 50
        self.y = GROUND_Y
        self.width = 60
        self.height = 70
        self.velocity_y = 0
        self.is_jumping = False
        self.is_ducking = False
        self.jump_speed = -18  # More powerful jump (was -15)
        self.gravity = 1     # Reduced gravity for higher jumps (was 1)
        self.shadow_offset = 10  # Shadow offset for visual effect
        
        # Load sprite images with fallbacks
        try:
            self.run_frames = [
                pygame.image.load(os.path.join(ASSETS_DIR, "dino_run1.png")),
                pygame.image.load(os.path.join(ASSETS_DIR, "dino_run2.png"))
            ]
            self.jump_image = pygame.image.load(os.path.join(ASSETS_DIR, "dino_jump.png"))
            self.duck_frames = [
                pygame.image.load(os.path.join(ASSETS_DIR, "dino_duck1.png")),
                pygame.image.load(os.path.join(ASSETS_DIR, "dino_duck2.png"))
            ]
            self.dead_image = pygame.image.load(os.path.join(ASSETS_DIR, "dino_dead.png"))
        except FileNotFoundError:
            print("Warning: Some dino image assets not found, using placeholders")
            # Create placeholder images
            self.run_frames = [
                create_placeholder_image(self.width, self.height, (50, 200, 50)),
                create_placeholder_image(self.width, self.height, (100, 200, 100))
            ]
            self.jump_image = create_placeholder_image(self.width, self.height, (50, 150, 200))
            self.duck_frames = [
                create_placeholder_image(self.width, self.height - 20, (200, 50, 50)),
                create_placeholder_image(self.width, self.height - 20, (200, 100, 100))
            ]
            self.dead_image = create_placeholder_image(self.width, self.height, (200, 50, 50))
        
        # Scale images to appropriate size
        for i in range(len(self.run_frames)):
            self.run_frames[i] = pygame.transform.scale(self.run_frames[i], (self.width, self.height))
        for i in range(len(self.duck_frames)):
            self.duck_frames[i] = pygame.transform.scale(self.duck_frames[i], (self.width, self.height - 20))
        self.jump_image = pygame.transform.scale(self.jump_image, (self.width, self.height))
        self.dead_image = pygame.transform.scale(self.dead_image, (self.width, self.height))
        
        self.frame_index = 0
        self.counter = 0
        self.animation_speed = 5

    def update(self):
        # Handle jumping physics
        if self.is_jumping:
            self.velocity_y += self.gravity
            self.y += self.velocity_y
            if self.y >= GROUND_Y:
                self.y = GROUND_Y
                self.is_jumping = False
                self.velocity_y = 0
                
        # Animation handling
        self.counter += 1
        if self.counter >= self.animation_speed:
            self.counter = 0
            self.frame_index = (self.frame_index + 1) % 2
                
        # Adjust height for ducking
        if self.is_ducking and not self.is_jumping:
            self.height = 50
        else:
            self.height = 70

    def draw(self, screen, game_over=False):
        # Choose the right sprite based on state
        if game_over:
            current_image = self.dead_image
        elif self.is_jumping:
            current_image = self.jump_image
        elif self.is_ducking:
            current_image = self.duck_frames[self.frame_index]
        else:
            current_image = self.run_frames[self.frame_index]
        
        # Draw shadow
        shadow = pygame.Surface((current_image.get_width(), 10)).convert_alpha()
        shadow.fill((0, 0, 0, 0))
        pygame.draw.ellipse(shadow, (0, 0, 0, 100), (0, 0, current_image.get_width(), 10))
        screen.blit(shadow, (self.x, GROUND_Y - 5))
        
        # Draw the dino
        screen.blit(current_image, (self.x, self.y - current_image.get_height()))

class Background:
    def __init__(self):
        # Load background elements with fallbacks
        try:
            self.sky = pygame.image.load(os.path.join(ASSETS_DIR, "sky.png"))
            self.mountains = pygame.image.load(os.path.join(ASSETS_DIR, "mountains.png"))
            self.clouds = pygame.image.load(os.path.join(ASSETS_DIR, "clouds.png"))
            self.ground = pygame.image.load(os.path.join(ASSETS_DIR, "ground.png"))
        except FileNotFoundError:
            print("Warning: Some background image assets not found, using placeholders")
            self.sky = create_placeholder_image(SCREEN_WIDTH, SCREEN_HEIGHT, (135, 206, 235))
            self.mountains = create_placeholder_image(SCREEN_WIDTH * 2, 200, (139, 137, 137))
            self.clouds = create_placeholder_image(SCREEN_WIDTH * 3, 100, (255, 255, 255))
            self.ground = create_placeholder_image(SCREEN_WIDTH * 2, 100, (210, 180, 140))
        
        # Scale images to fit screen
        self.sky = pygame.transform.scale(self.sky, (SCREEN_WIDTH, SCREEN_HEIGHT))
        self.mountains = pygame.transform.scale(self.mountains, (SCREEN_WIDTH * 2, 200))
        self.clouds = pygame.transform.scale(self.clouds, (SCREEN_WIDTH * 3, 100))
        self.ground = pygame.transform.scale(self.ground, (SCREEN_WIDTH * 2, 100))
        
        # Track positions for parallax scrolling
        self.mountains_pos = 0
        self.clouds_pos = 0
        self.ground_pos = 0
    
    def update(self, speed):
        # Move background elements at different speeds for parallax effect
        self.mountains_pos -= speed * 0.2
        self.clouds_pos -= speed * 0.5
        self.ground_pos -= speed
        
        # Reset positions when they go off screen
        if self.mountains_pos <= -SCREEN_WIDTH:
            self.mountains_pos = 0
        if self.clouds_pos <= -SCREEN_WIDTH:
            self.clouds_pos = 0
        if self.ground_pos <= -SCREEN_WIDTH:
            self.ground_pos = 0
    
    def draw(self, screen):
        # Draw background layers in order
        screen.blit(self.sky, (0, 0))
        screen.blit(self.mountains, (self.mountains_pos, SCREEN_HEIGHT - 250))
        screen.blit(self.mountains, (self.mountains_pos + SCREEN_WIDTH, SCREEN_HEIGHT - 250))
        screen.blit(self.clouds, (self.clouds_pos, 50))
        screen.blit(self.clouds, (self.clouds_pos + SCREEN_WIDTH, 50))
        screen.blit(self.ground, (self.ground_pos, GROUND_Y - 50))
        screen.blit(self.ground, (self.ground_pos + SCREEN_WIDTH, GROUND_Y - 50))

class Obstacle:
    def __init__(self, x, speed):
        self.x = x
        self.y = GROUND_Y
        self.speed = speed
        
        # Randomly choose cactus type
        cactus_type = random.randint(1, 3)
        try:
            self.image = pygame.image.load(os.path.join(ASSETS_DIR, f"cactus{cactus_type}.png"))
        except FileNotFoundError:
            self.width = 40
            self.height = random.randint(40, 70)
            self.image = create_placeholder_image(self.width, self.height, (0, 100, 0))
            return
            
        self.width = 40
        self.height = random.randint(40, 70)
        self.image = pygame.transform.scale(self.image, (self.width, self.height))

    def update(self):
        self.x -= self.speed

    def draw(self, screen):
        # Draw shadow
        shadow = pygame.Surface((self.width, 10)).convert_alpha()
        shadow.fill((0, 0, 0, 0))
        pygame.draw.ellipse(shadow, (0, 0, 0, 100), (0, 0, self.width, 10))
        screen.blit(shadow, (self.x, GROUND_Y - 5))
        
        # Draw cactus
        screen.blit(self.image, (self.x, self.y - self.height))

    def off_screen(self):
        return self.x + self.width < 0

    def collides_with(self, dino):
        # Create slightly smaller collision boxes for better gameplay feel
        padding = 10
        dino_rect = pygame.Rect(dino.x + padding, dino.y - dino.height + padding, 
                              dino.width - 2*padding, dino.height - padding)
        obs_rect = pygame.Rect(self.x + padding, self.y - self.height + padding, 
                              self.width - 2*padding, self.height - padding)
        return dino_rect.colliderect(obs_rect)

class Crow:
    def __init__(self, x, speed):
        self.x = x
        self.width = 50
        self.height = 40
        # Place crows at standing dino height to ensure collisions when not ducking
        self.bottom = random.randint(GROUND_Y - 70, GROUND_Y - 40)  # Better height for collision
        self.speed = speed
        
        # Load bird animation frames with fallbacks
        try:
            self.frames = [
                pygame.image.load(os.path.join(ASSETS_DIR, "bird1.png")),
                pygame.image.load(os.path.join(ASSETS_DIR, "bird2.png"))
            ]
        except FileNotFoundError:
            print("Warning: Bird image assets not found, using placeholders")
            self.frames = [
                create_placeholder_image(self.width, self.height, (0, 0, 0)),
                create_placeholder_image(self.width, self.height, (50, 50, 50))
            ]
        
        # Scale images
        for i in range(len(self.frames)):
            self.frames[i] = pygame.transform.scale(self.frames[i], (self.width, self.height))
            
        self.frame_index = 0
        self.counter = 0
        self.animation_speed = 8

    def update(self):
        self.x -= self.speed
        
        # Animation
        self.counter += 1
        if self.counter >= self.animation_speed:
            self.counter = 0
            self.frame_index = (self.frame_index + 1) % 2

    def draw(self, screen):
        # Draw shadow (fainter and smaller since bird is in the air)
        shadow = pygame.Surface((self.width * 0.8, 6)).convert_alpha()
        shadow.fill((0, 0, 0, 0))
        pygame.draw.ellipse(shadow, (0, 0, 0, 60), (0, 0, self.width * 0.8, 6))
        screen.blit(shadow, (self.x + 5, GROUND_Y - 5))
        
        # Draw bird
        screen.blit(self.frames[self.frame_index], (self.x, self.bottom - self.height))

    def off_screen(self):
        return self.x + self.width < 0

    def collides_with(self, dino):
        # Reduce padding to make collision detection more accurate
        padding = 5
        
        # Create dino collision box based on ducking state
        if dino.is_ducking:
            # Smaller collision box when ducking
            dino_rect = pygame.Rect(
                dino.x + padding, 
                dino.y - dino.height + padding + 20,  # Higher y when ducking (smaller height) 
                dino.width - 2*padding, 
                dino.height - padding - 20
            )
        else:
            # Normal collision box when standing/running
            dino_rect = pygame.Rect(
                dino.x + padding, 
                dino.y - dino.height + padding,
                dino.width - 2*padding, 
                dino.height - padding
            )
        
        # Crow collision box
        crow_rect = pygame.Rect(
            self.x + padding, 
            self.bottom - self.height + padding, 
            self.width - 2*padding, 
            self.height - padding
        )
        
        # Debug visualization - uncomment if needed to see collision boxes
        # pygame.draw.rect(screen, (255, 0, 0, 128), dino_rect, 2)
        # pygame.draw.rect(screen, (0, 255, 0, 128), crow_rect, 2)
        
        return dino_rect.colliderect(crow_rect)

class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.velocity_x = random.uniform(-2, 2)
        self.velocity_y = random.uniform(-5, -1)
        self.size = random.randint(2, 6)
        self.lifetime = random.randint(20, 40)

    def update(self):
        self.x += self.velocity_x
        self.y += self.velocity_y
        self.velocity_y += 0.2  # Gravity
        self.lifetime -= 1
        self.size = max(0, self.size - 0.1)
        
    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), int(self.size))

class ParticleSystem:
    def __init__(self):
        self.particles = []
        
    def add_particles(self, x, y, color, count=10):
        for _ in range(count):
            self.particles.append(Particle(x, y, color))
            
    def update(self):
        self.particles = [p for p in self.particles if p.lifetime > 0]
        for p in self.particles:
            p.update()
            
    def draw(self, screen):
        for p in self.particles:
            p.draw(screen)

# Load sounds function
def load_sounds():
    sounds = {}
    try:
        mixer.init()
        sounds['jump'] = mixer.Sound(os.path.join(ASSETS_DIR, "jump.wav"))
        sounds['die'] = mixer.Sound(os.path.join(ASSETS_DIR, "die.wav"))
        sounds['point'] = mixer.Sound(os.path.join(ASSETS_DIR, "point.wav"))
        
        # Set volume
        for sound in sounds.values():
            sound.set_volume(0.5)
    except:
        # Fallback if sound loading fails
        print("Warning: Could not load sounds")
    
    return sounds

def create_placeholder_image(width, height, color=(100, 100, 255)):
    """Create a simple placeholder image when asset files are missing."""
    surface = pygame.Surface((width, height), pygame.SRCALPHA)
    pygame.draw.rect(surface, color, (0, 0, width, height))
    # Add an outline
    pygame.draw.rect(surface, (0, 0, 0), (0, 0, width, height), 2)
    return surface

# ---------------------------
# Main game loop using Pygame
# ---------------------------
def main():
    global action
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Dino Runner")
    clock = pygame.time.Clock()
    
    # Set up fonts
    font = pygame.font.Font(None, 36)
    large_font = pygame.font.Font(None, 72)
    
    # Create game objects
    dino = Dino()
    background = Background()
    obstacles = []
    particle_system = ParticleSystem()
    
    # Load sounds if available
    try:
        sounds = load_sounds()
    except:
        sounds = {}
    
    # Game state variables
    spawn_timer = 0
    score = 0
    high_score = 0
    game_over = False
    base_speed = 5
    milestone_interval = 500  # Score interval for speed increase and sound
    last_milestone = 0
    screen_shake = 0  # For death effect
    
    # Try to load high score
    try:
        if os.path.exists("highscore.txt"):
            with open("highscore.txt", "r") as f:
                high_score = int(f.read().strip())
    except:
        high_score = 0

    while True:
        clock.tick(FPS)
        
        # Apply screen shake effect
        shake_offset = [0, 0]
        if screen_shake > 0:
            shake_offset = [random.randint(-5, 5), random.randint(-5, 5)]
            screen_shake -= 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Save high score before exiting
                try:
                    with open("highscore.txt", "w") as f:
                        f.write(str(high_score))
                except:
                    pass
                pygame.quit()
                sys.exit()
            
            # Handle restart on game over
            if game_over and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Reset game
                    dino = Dino()
                    obstacles = []
                    spawn_timer = 0
                    score = 0
                    game_over = False
                    screen_shake = 0
                    base_speed = 5

        current_speed = base_speed + (score // 500) * 0.5
        
        # Update background with parallax scrolling
        if not game_over:
            background.update(current_speed)

        # Handle jump action from pose detection
        if not game_over:
            if action == "jump" and not dino.is_jumping:
                dino.is_jumping = True
                dino.velocity_y = dino.jump_speed
                if 'jump' in sounds:
                    sounds['jump'].play()
                # Add dust particles when jumping
                particle_system.add_particles(dino.x + dino.width // 2, GROUND_Y, (139, 119, 101), 5)
                
            # Simplified duck logic - can duck even while in the air like in original game
            dino.is_ducking = (action == "duck")
            
            # Update dino
            dino.update()
            
            # Update particles
            particle_system.update()
            
            # Check for score milestone
            if score % milestone_interval == 0 and score > 0 and score != last_milestone:
                if 'point' in sounds:
                    sounds['point'].play()
                last_milestone = score
                # Add celebratory particles
                particle_system.add_particles(
                    700, 50, (255, 215, 0), 20
                )

            # Spawn obstacles
            spawn_timer += 1
            # Increase base timer from 60 to 90 for more space between obstacles
            if spawn_timer > 90 - min(score // 1000, 30):  # Slower spawning
                if random.random() < 0.3:  # 30% chance for crow
                    obstacles.append(Crow(SCREEN_WIDTH, current_speed))
                else:  # 70% chance for cactus
                    obstacles.append(Obstacle(SCREEN_WIDTH, current_speed))
                spawn_timer = 0
            
            # Update obstacles
            for obs in obstacles:
                obs.update()
                if obs.collides_with(dino):
                    game_over = True
                    if 'die' in sounds:
                        sounds['die'].play()
                    screen_shake = 15  # Start screen shake
                    
                    # Add explosion particles
                    particle_system.add_particles(
                        dino.x + dino.width // 2,
                        dino.y - dino.height // 2,
                        (255, 0, 0), 30
                    )
                    
                    # Update high score if needed
                    if score > high_score:
                        high_score = score
            
            # Remove obstacles that are off screen
            obstacles = [obs for obs in obstacles if not obs.off_screen()]
            
            # Increment score
            score += 1

        # Drawing
        # Start with background
        background.draw(screen)
        
        # Draw ground line (in case background image doesn't have it)
        pygame.draw.line(screen, (83, 83, 83, 128), 
                         (0, GROUND_Y), (SCREEN_WIDTH, GROUND_Y), 2)
        
        # Draw obstacles
        for obs in obstacles:
            obs.draw(screen)
        
        # Draw particles
        particle_system.draw(screen)
        
        # Draw dino
        dino.draw(screen, game_over)
        
        # Draw UI - score and high score
        score_text = font.render(f"Score: {score}", True, (83, 83, 83))
        hi_score_text = font.render(f"HI: {high_score}", True, (83, 83, 83))
        screen.blit(score_text, (20, 20))
        screen.blit(hi_score_text, (SCREEN_WIDTH - 150, 20))
        
        # Draw game over screen
        if game_over:
            # Semi-transparent overlay
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            screen.blit(overlay, (0, 0))
            
            # Game over text
            game_over_text = large_font.render("GAME OVER", True, (255, 255, 255))
            screen.blit(game_over_text, (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, 
                                      SCREEN_HEIGHT // 2 - 50))
            
            # Instructions to restart
            restart_text = font.render("Press SPACE to restart", True, (255, 255, 255))
            screen.blit(restart_text, (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, 
                                     SCREEN_HEIGHT // 2 + 10))
            
            # Final score display
            final_score_text = font.render(f"Your Score: {score}", True, (255, 255, 255))
            screen.blit(final_score_text, (SCREEN_WIDTH // 2 - final_score_text.get_width() // 2, 
                                        SCREEN_HEIGHT // 2 + 50))

        # Apply screen shake
        if shake_offset != [0, 0]:
            buffer_surface = screen.copy()
            screen.fill((0, 0, 0))
            screen.blit(buffer_surface, shake_offset)
            
        pygame.display.flip()

# ---------------------------
# Start the pose detection thread and launch the game
# ---------------------------
if __name__ == "__main__":
    # Create assets directory if it doesn't exist
    os.makedirs(ASSETS_DIR, exist_ok=True)
    
    # Create asset files if they don't exist yet
    # This is a placeholder - you'll need to create or download these assets
    
    cv_thread = threading.Thread(target=pose_detection, daemon=True)
    cv_thread.start()
    main()