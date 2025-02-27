import pygame

class SpriteSheet:
    """
    Loads a single sprite sheet and provides a helper method to extract sub-images.
    """
    def __init__(self, filename):
        # Load the sprite sheet (PNG with alpha)
        self.sheet = pygame.image.load(filename).convert_alpha()

    def get_image(self, x, y, width, height):
        """
        Extract a sub-image from (x, y, width, height) on the sprite sheet.
        Returns a new Surface with that sub-image.
        """
        image = pygame.Surface((width, height), pygame.SRCALPHA)
        image.blit(self.sheet, (0, 0), (x, y, width, height))
        return image

# ----------------------------------------------------------------
# Dino class (sprite-based)
# ----------------------------------------------------------------
class Dino:
    def __init__(self, sprite_sheet):
        """
        Example usage of frames:
          - self.run_frames: 2 frames for running
          - self.duck_frames: 2 frames for ducking
          - self.jump_frame: single frame for jumping
        Adjust coordinates/width/height to match your sheet.
        """
        # Position & physics
        self.x = 50
        self.y = 300
        self.width = 40    # for collision bounding box
        self.height = 50   # for collision bounding box
        self.velocity_y = 0
        self.is_jumping = False
        self.is_ducking = False
        self.jump_speed = -15
        self.gravity = 1

        # Animation
        self.run_frames = [
            sprite_sheet.get_image(1338, 2, 44, 47),  # Dino run 1
            sprite_sheet.get_image(1392, 2, 44, 47),  # Dino run 2
        ]
        self.duck_frames = [
            sprite_sheet.get_image(1866, 19, 59, 28), # Dino duck 1
            sprite_sheet.get_image(1933, 19, 59, 28), # Dino duck 2
        ]
        self.jump_frame = sprite_sheet.get_image(446, 2, 44, 47)  # placeholder

        self.anim_index = 0
        self.anim_timer = 0.0
        self.anim_speed = 0.15  # seconds between frames
        self.current_image = self.run_frames[0]

    def update(self):
        """
        Same logic as your rectangle-based version, but we also pick the correct frame.
        """
        # Jump physics
        if self.is_jumping:
            self.velocity_y += self.gravity
            self.y += self.velocity_y
            if self.y >= 300:
                self.y = 300
                self.is_jumping = False
                self.velocity_y = 0

        # Adjust collision box if ducking (just like your old code).
        if self.is_ducking and not self.is_jumping:
            self.height = 30
        else:
            self.height = 50

        # Decide which frames to use
        if self.is_jumping:
            # Single jump frame
            self.current_image = self.jump_frame
        elif self.is_ducking:
            # Cycle duck frames
            self.anim_timer += self.anim_speed
            if self.anim_timer >= 1.0:
                self.anim_timer = 0
                self.anim_index = (self.anim_index + 1) % len(self.duck_frames)
            self.current_image = self.duck_frames[self.anim_index]
        else:
            # Cycle run frames
            self.anim_timer += self.anim_speed
            if self.anim_timer >= 1.0:
                self.anim_timer = 0
                self.anim_index = (self.anim_index + 1) % len(self.run_frames)
            self.current_image = self.run_frames[self.anim_index]

    def draw(self, screen):
        """
        Blit the current image instead of drawing a rectangle.
        """
        # The top-left corner of the sprite’s position
        # If your sprite is drawn from bottom-left, adjust accordingly
        y_draw = self.y - self.current_image.get_height()
        screen.blit(self.current_image, (self.x, y_draw))


# ----------------------------------------------------------------
# Obstacle class (sprite-based cactus)
# ----------------------------------------------------------------
class Obstacle:
    """
    A ground-based cactus. In your old code, you had random heights.
    With sprite-based cacti, you can store multiple frames for variety.
    """
    def __init__(self, sprite_sheet, x, speed):
        self.x = x
        self.y = 300  # GROUND_Y in your code
        self.speed = speed

        # For collision bounding box
        self.width = 20
        self.height = 50

        # For variety, you might have multiple frames. Let's pick one randomly:
        # (Placeholder coords)
        cactus_frames = [
            sprite_sheet.get_image(0, 0, 20, 50),
            sprite_sheet.get_image(25, 0, 25, 50),
        ]
        import random
        self.image = random.choice(cactus_frames)
        self.rect = self.image.get_rect()

    def update(self):
        self.x -= self.speed

    def draw(self, screen):
        # Position the cactus so its bottom aligns with the ground line (y=300).
        y_draw = self.y - self.image.get_height()
        screen.blit(self.image, (self.x, y_draw))

    def off_screen(self):
        return (self.x + self.image.get_width()) < 0

    def collides_with(self, dino):
        """
        For collision, we can still use bounding boxes:
        dino.x, dino.y, dino.width, dino.height
        """
        dino_rect = pygame.Rect(dino.x, dino.y - dino.height, dino.width, dino.height)
        obs_rect = pygame.Rect(self.x, self.y - self.image.get_height(),
                               self.image.get_width(), self.image.get_height())
        return dino_rect.colliderect(obs_rect)


# ----------------------------------------------------------------
# Crow class (sprite-based)
# ----------------------------------------------------------------
class Crow:
    """
    A flying obstacle. We have two frames for flapping wings.
    """
    def __init__(self, sprite_sheet, x, speed):
        self.x = x
        self.speed = speed
        self.bottom = 260  # same logic from your code

        # For collision bounding box
        self.width = 30
        self.height = 20

        # Bird frames (placeholder coords)
        self.frames = [
            sprite_sheet.get_image(260, 2, 46, 40),
            sprite_sheet.get_image(312, 2, 46, 40),
        ]
        self.index = 0
        self.timer = 0.0
        self.anim_speed = 0.15
        self.current_image = self.frames[0]

    def update(self):
        # Animate
        self.timer += self.anim_speed
        if self.timer >= 1.0:
            self.timer = 0
            self.index = (self.index + 1) % len(self.frames)
            self.current_image = self.frames[self.index]

        # Move left
        self.x -= self.speed

    def draw(self, screen):
        top = self.bottom - self.current_image.get_height()
        screen.blit(self.current_image, (self.x, top))

    def off_screen(self):
        return (self.x + self.current_image.get_width()) < 0

    def collides_with(self, dino):
        dino_rect = pygame.Rect(dino.x, dino.y - dino.height, dino.width, dino.height)
        crow_rect = pygame.Rect(self.x,
                                self.bottom - self.current_image.get_height(),
                                self.current_image.get_width(),
                                self.current_image.get_height())
        return dino_rect.colliderect(crow_rect)
