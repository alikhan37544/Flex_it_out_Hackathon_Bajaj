import cv2
import mediapipe as mp
import pygame
import random
import math
import os
import time
from pygame import gfxdraw
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

class SliceEffect:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.lifetime = 0.5  # Effect lasts for 0.5 seconds
        self.time_alive = 0
        self.particles = []
        
        # Create particles
        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(50, 150)
            size = random.randint(2, 5)
            self.particles.append({
                'x': x,
                'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'size': size,
                'alpha': 255,
            })
    
    def update(self, dt):
        self.time_alive += dt
        for p in self.particles:
            p['x'] += p['vx'] * dt
            p['y'] += p['vy'] * dt
            p['alpha'] = int(255 * (1 - self.time_alive / self.lifetime))
    
    def draw(self, screen):
        for p in self.particles:
            if p['alpha'] > 0:
                color_with_alpha = (*self.color, p['alpha'])
                gfxdraw.filled_circle(screen, int(p['x']), int(p['y']), p['size'], color_with_alpha)
    
    def is_expired(self):
        return self.time_alive >= self.lifetime

class GlitterEffect:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.lifetime = 1.5  # Effect lasts for 1.5 seconds
        self.time_alive = 0
        self.particles = []
        self.colors = [
            (255, 215, 0),   # Gold
            (255, 255, 255), # White
            (255, 105, 180), # Hot Pink
            (0, 191, 255),   # Deep Sky Blue
            (50, 205, 50)    # Lime Green
        ]
        
        # Create particles
        for _ in range(50):  # More particles for a richer effect
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(50, 200)
            size = random.uniform(1, 4)
            color = random.choice(self.colors)
            self.particles.append({
                'x': x,
                'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'size': size,
                'alpha': 255,
                'color': color,
                'decay_rate': random.uniform(0.8, 1.5),  # Different decay rates
                'rotation': random.uniform(0, 360)
            })
    
    def update(self, dt):
        self.time_alive += dt
        for p in self.particles:
            p['x'] += p['vx'] * dt
            p['y'] += p['vy'] * dt
            # Add gravity
            p['vy'] += 50 * dt
            # Decay alpha faster for shorter-lived particles
            p['alpha'] = max(0, int(255 * (1 - (self.time_alive * p['decay_rate'] / self.lifetime))))
            # Rotate the particle
            p['rotation'] += dt * 180  # 180 degrees per second
    
    def draw(self, screen):
        for p in self.particles:
            if p['alpha'] > 0:
                color_with_alpha = (*p['color'], p['alpha'])
                # Draw a small rectangle with rotation for more interesting shape
                size = int(p['size'] * 2)
                particle_surface = pygame.Surface((size, size), pygame.SRCALPHA)
                pygame.draw.rect(particle_surface, color_with_alpha, (0, 0, size, size))
                rotated = pygame.transform.rotate(particle_surface, p['rotation'])
                pos = (int(p['x'] - rotated.get_width()/2), int(p['y'] - rotated.get_height()/2))
                screen.blit(rotated, pos)
                
                # Add a glow effect
                glow_radius = int(p['size'] * 2)
                glow_surface = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
                glow_color = (*p['color'], p['alpha']//4)
                pygame.draw.circle(glow_surface, glow_color, (glow_radius, glow_radius), glow_radius)
                glow_pos = (int(p['x'] - glow_radius), int(p['y'] - glow_radius))
                screen.blit(glow_surface, glow_pos)
    
    def is_expired(self):
        return self.time_alive >= self.lifetime

class FruitBase:
    def __init__(self, x, y, radius, color, vx, vy):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.vx = vx  # Horizontal velocity (pixels/second)
        self.vy = vy  # Vertical velocity (pixels/second)
        self.sliced = False
        self.is_bomb = False
        self.rotation = 0
        self.rotation_speed = random.uniform(-180, 180)  # Degrees per second
        self.slice_time = None
        self.shadow_offset = 15  # Shadow offset for 3D effect
        
    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        gravity = 500  # pixels per second^2
        self.vy += gravity * dt
        self.rotation += self.rotation_speed * dt
        
        # If sliced, add some "explosion" effect by modifying velocities
        if self.sliced and self.slice_time is None:
            self.slice_time = 0
            # Add random impulse to sliced fruit
            self.vx += random.uniform(-100, 100)
            self.vy -= random.uniform(0, 100)
            self.rotation_speed *= 2  # Rotate faster when sliced

    def draw(self, screen):
        # Abstract method to be implemented by subclasses
        pass
    
    def draw_shadow(self, screen):
        # Draw shadow - slightly larger, darker and offset
        shadow_surface = pygame.Surface((self.radius*2 + 10, self.radius*2 + 10), pygame.SRCALPHA)
        shadow_color = (0, 0, 0, 100)  # Semi-transparent black
        pygame.draw.circle(shadow_surface, shadow_color, (self.radius + 5, self.radius + 5), self.radius)
        # Apply gaussian blur (simplified with scaling)
        shadow_surface = pygame.transform.smoothscale(shadow_surface, (int(self.radius*2.2), int(self.radius*2.2)))
        shadow_surface = pygame.transform.smoothscale(shadow_surface, (int(self.radius*2.4), int(self.radius*2.4)))
        screen.blit(shadow_surface, (self.x - self.radius*1.2 + self.shadow_offset, self.y - self.radius*1.2 + self.shadow_offset))

class Fruit(FruitBase):
    FRUIT_TYPES = [
        {"name": "watermelon", "color": (0, 180, 0), "inner_color": (220, 60, 80)},
        {"name": "orange", "color": (255, 140, 0), "inner_color": (255, 190, 80)},
        {"name": "apple", "color": (220, 0, 0), "inner_color": (255, 240, 240)},
        {"name": "pineapple", "color": (200, 180, 0), "inner_color": (255, 240, 150)},
        {"name": "kiwi", "color": (100, 180, 0), "inner_color": (180, 210, 60)},
    ]
    
    def __init__(self, x, y, radius, color, vx, vy):
        super().__init__(x, y, radius, color, vx, vy)
        fruit_type = random.choice(self.FRUIT_TYPES)
        self.color = fruit_type["color"]
        self.inner_color = fruit_type["inner_color"]
        self.name = fruit_type["name"]
        self.points = random.randint(1, 3)  # Different fruits have different point values
        
        # Sliced halves properties
        self.left_half_x = self.x
        self.left_half_y = self.y
        self.right_half_x = self.x
        self.right_half_y = self.y
        self.left_vx = 0
        self.left_vy = 0
        self.right_vx = 0
        self.right_vy = 0

    def update(self, dt):
        super().update(dt)
        
        # Update sliced halves
        if self.sliced:
            if self.slice_time is not None and self.slice_time == 0:
                # Initialize the slice dynamics
                self.left_half_x = self.x
                self.left_half_y = self.y
                self.right_half_x = self.x
                self.right_half_y = self.y
                
                # Give the halves opposing horizontal velocities
                self.left_vx = self.vx - random.uniform(50, 150)
                self.right_vx = self.vx + random.uniform(50, 150)
                
                # Both halves should have similar but slightly different vertical velocities
                self.left_vy = self.vy - random.uniform(20, 50)
                self.right_vy = self.vy - random.uniform(20, 50)
                
                self.slice_time += dt
            elif self.slice_time is not None:
                # Update positions of the halves
                self.left_half_x += self.left_vx * dt
                self.left_half_y += self.left_vy * dt
                self.right_half_x += self.right_vx * dt
                self.right_half_y += self.right_vy * dt
                
                # Apply gravity to the halves
                gravity = 500
                self.left_vy += gravity * dt
                self.right_vy += gravity * dt
                
                self.slice_time += dt

    def draw(self, screen):
        if not self.sliced:
            # Draw shadow for 3D effect
            self.draw_shadow(screen)
            
            # Draw unsliced fruit
            surface = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
            pygame.draw.circle(surface, self.color, (self.radius, self.radius), self.radius)
            
            # Add a shine effect (gradient)
            shine_radius = int(self.radius * 0.7)
            shine_pos = (int(self.radius * 0.7), int(self.radius * 0.7))
            for r in range(shine_radius, 0, -1):
                alpha = 150 - int(150 * (r / shine_radius))
                pygame.draw.circle(surface, (255, 255, 255, alpha), shine_pos, r)
            
            # Rotate the fruit
            rotated_surface = pygame.transform.rotate(surface, self.rotation)
            new_rect = rotated_surface.get_rect(center=(self.x, self.y))
            screen.blit(rotated_surface, new_rect.topleft)
        else:
            # Draw the sliced halves
            self.draw_sliced_halves(screen)
    
    def draw_sliced_halves(self, screen):
        # Draw shadows for the halves
        shadow_color = (0, 0, 0, 80)
        
        # Draw left half shadow
        left_shadow = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
        pygame.draw.circle(left_shadow, shadow_color, (self.radius, self.radius), self.radius)
        # Cut the right side off
        pygame.draw.rect(left_shadow, (0, 0, 0, 0), (self.radius, 0, self.radius, self.radius*2))
        rotated_shadow = pygame.transform.rotate(left_shadow, self.rotation)
        shadow_rect = rotated_shadow.get_rect(center=(self.left_half_x + self.shadow_offset, self.left_half_y + self.shadow_offset))
        screen.blit(rotated_shadow, shadow_rect.topleft)
        
        # Draw right half shadow
        right_shadow = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
        pygame.draw.circle(right_shadow, shadow_color, (self.radius, self.radius), self.radius)
        # Cut the left side off
        pygame.draw.rect(right_shadow, (0, 0, 0, 0), (0, 0, self.radius, self.radius*2))
        rotated_shadow = pygame.transform.rotate(right_shadow, self.rotation)
        shadow_rect = rotated_shadow.get_rect(center=(self.right_half_x + self.shadow_offset, self.right_half_y + self.shadow_offset))
        screen.blit(rotated_shadow, shadow_rect.topleft)
        
        # Draw left half
        left_half = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
        pygame.draw.circle(left_half, self.color, (self.radius, self.radius), self.radius)
        # Draw the inner part (flesh) of the fruit
        pygame.draw.circle(left_half, self.inner_color, (self.radius, self.radius), self.radius * 0.9)
        # Cut the right side off
        pygame.draw.rect(left_half, (0, 0, 0, 0), (self.radius, 0, self.radius, self.radius*2))
        # Add slice line
        pygame.draw.line(left_half, (255, 255, 255, 200), (self.radius, 0), (self.radius, self.radius*2), 2)
        rotated_left = pygame.transform.rotate(left_half, self.rotation)
        left_rect = rotated_left.get_rect(center=(self.left_half_x, self.left_half_y))
        screen.blit(rotated_left, left_rect.topleft)
        
        # Draw right half
        right_half = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
        pygame.draw.circle(right_half, self.color, (self.radius, self.radius), self.radius)
        # Draw the inner part (flesh) of the fruit
        pygame.draw.circle(right_half, self.inner_color, (self.radius, self.radius), self.radius * 0.9)
        # Cut the left side off
        pygame.draw.rect(right_half, (0, 0, 0, 0), (0, 0, self.radius, self.radius*2))
        # Add slice line
        pygame.draw.line(right_half, (255, 255, 255, 200), (self.radius, 0), (self.radius, self.radius*2), 2)
        rotated_right = pygame.transform.rotate(right_half, self.rotation)
        right_rect = rotated_right.get_rect(center=(self.right_half_x, self.right_half_y))
        screen.blit(rotated_right, right_rect.topleft)

class Bomb(FruitBase):
    def __init__(self, x, y, radius, color, vx, vy):
        super().__init__(x, y, radius, color, vx, vy)
        self.is_bomb = True
        self.fuse_length = random.uniform(0.5, 0.9)  # Normalized fuse length (0-1)
        self.fuse_burn_speed = random.uniform(2, 4)  # Degrees per second
        self.fuse_burn_pos = 0  # Current burn position
        self.warning_pulse = 0  # For pulsating warning effect
        
    def update(self, dt):
        super().update(dt)
        
        # Animate the fuse burning
        if not self.sliced:
            self.fuse_burn_pos += self.fuse_burn_speed * dt
            self.fuse_burn_pos = min(self.fuse_burn_pos, self.fuse_length)
            
            # Update warning pulse
            self.warning_pulse += dt * 5  # Speed of pulse
            if self.warning_pulse > 2 * math.pi:
                self.warning_pulse -= 2 * math.pi
        
    def draw(self, screen):
        # Draw shadow for 3D effect
        self.draw_shadow(screen)
        
        if self.sliced:
            # Draw explosion effect
            for _ in range(5):
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(0, self.radius * 1.5)
                size = random.randint(3, 8)
                x = self.x + math.cos(angle) * distance
                y = self.y + math.sin(angle) * distance
                color_value = random.randint(200, 255)
                color = (color_value, color_value * 0.6, 0)
                pygame.draw.circle(screen, color, (int(x), int(y)), size)
            
            # Draw smoke particles
            for _ in range(10):
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(0, self.radius * 2)
                size = random.randint(2, 6)
                x = self.x + math.cos(angle) * distance
                y = self.y + math.sin(angle) * distance
                gray_value = random.randint(100, 200)
                color = (gray_value, gray_value, gray_value, random.randint(100, 180))
                gfxdraw.filled_circle(screen, int(x), int(y), size, color)
        else:
            # Draw bomb body
            surface = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
            
            # Draw the main bomb body
            pygame.draw.circle(surface, (20, 20, 20), (self.radius, self.radius), self.radius)
            
            # Draw the fuse cap at the top
            cap_width = self.radius * 0.6
            cap_height = self.radius * 0.3
            pygame.draw.rect(surface, (80, 80, 80), 
                           (self.radius - cap_width/2, 
                            self.radius - self.radius - cap_height/2,
                            cap_width, cap_height), 
                           border_radius=int(cap_height/2))
            
            # Draw the fuse
            fuse_start = (self.radius, self.radius - self.radius * 0.8)
            fuse_end = (self.radius + self.radius * 0.5 * self.fuse_length, 
                       self.radius - self.radius * 1.3)
            
            # Draw the unburnt part of the fuse
            if self.fuse_burn_pos < self.fuse_length:
                unburnt_end = (
                    self.radius + self.radius * 0.5 * self.fuse_burn_pos,
                    self.radius - self.radius * (1.3 - 0.3 * self.fuse_burn_pos / self.fuse_length)
                )
                pygame.draw.line(surface, (150, 150, 150), unburnt_end, fuse_end, 3)
            
            # Draw the burnt part of the fuse
            if self.fuse_burn_pos > 0:
                burnt_end = (
                    self.radius + self.radius * 0.5 * min(self.fuse_burn_pos, self.fuse_length),
                    self.radius - self.radius * (1.3 - 0.3 * min(self.fuse_burn_pos, self.fuse_length) / self.fuse_length)
                )
                pygame.draw.line(surface, (255, 100, 0), fuse_start, burnt_end, 3)
            
            # Add highlight to bomb
            highlight_pos = (int(self.radius * 0.7), int(self.radius * 0.7))
            highlight_radius = int(self.radius * 0.3)
            for r in range(highlight_radius, 0, -1):
                alpha = 100 - int(100 * (r / highlight_radius))
                pygame.draw.circle(surface, (100, 100, 100, alpha), highlight_pos, r)
            
            # Add warning pulsing ring
            warning_alpha = int(127 + 127 * math.sin(self.warning_pulse))
            warning_color = (255, 0, 0, warning_alpha)
            gfxdraw.aacircle(surface, self.radius, self.radius, self.radius + 3, warning_color)
            
            # Rotate and draw the bomb
            rotated_surface = pygame.transform.rotate(surface, self.rotation)
            new_rect = rotated_surface.get_rect(center=(self.x, self.y))
            screen.blit(rotated_surface, new_rect.topleft)

class PowerUp(FruitBase):
    def __init__(self, x, y, radius, power_type, vx, vy):
        color = (120, 200, 255) if power_type == "freeze" else (255, 215, 0)
        super().__init__(x, y, radius, color, vx, vy)
        self.power_type = power_type  # "freeze" or "bonus"
        self.glow_intensity = 0
        self.glow_direction = 1  # 1 for increasing, -1 for decreasing
    
    def update(self, dt):
        super().update(dt)
        
        # Animate the glow
        self.glow_intensity += dt * 2 * self.glow_direction
        if self.glow_intensity > 1:
            self.glow_intensity = 1
            self.glow_direction = -1
        elif self.glow_intensity < 0:
            self.glow_intensity = 0
            self.glow_direction = 1
    
    def draw(self, screen):
        # Draw shadow
        self.draw_shadow(screen)
        
        # Create surface for power-up
        surface = pygame.Surface((self.radius*2.5, self.radius*2.5), pygame.SRCALPHA)
        
        # Draw outer glow
        glow_radius = int(self.radius * (1.2 + 0.3 * self.glow_intensity))
        glow_color = list(self.color) + [100 + int(100 * self.glow_intensity)]
        pygame.draw.circle(surface, glow_color, (int(surface.get_width()/2), int(surface.get_height()/2)), glow_radius)
        
        # Draw inner circle
        pygame.draw.circle(surface, self.color, (int(surface.get_width()/2), int(surface.get_height()/2)), self.radius)
        
        # Draw icon based on power type
        if self.power_type == "freeze":
            # Draw a snowflake
            center_x, center_y = int(surface.get_width()/2), int(surface.get_height()/2)
            icon_size = int(self.radius * 0.7)
            
            # Draw 6 lines radiating from center
            for i in range(6):
                angle = i * math.pi / 3
                end_x = center_x + math.cos(angle) * icon_size
                end_y = center_y + math.sin(angle) * icon_size
                pygame.draw.line(surface, (255, 255, 255), (center_x, center_y), (end_x, end_y), 2)
                
                # Draw small lines at angles
                for j in range(2):
                    sub_angle = angle + (j*2-1) * math.pi/6
                    sub_start_x = center_x + math.cos(angle) * icon_size * 0.5
                    sub_start_y = center_y + math.sin(angle) * icon_size * 0.5
                    sub_end_x = sub_start_x + math.cos(sub_angle) * icon_size * 0.3
                    sub_end_y = sub_start_y + math.sin(sub_angle) * icon_size * 0.3
                    pygame.draw.line(surface, (255, 255, 255), (sub_start_x, sub_start_y), (sub_end_x, sub_end_y), 2)
        
        elif self.power_type == "bonus":
            # Draw a star
            center_x, center_y = int(surface.get_width()/2), int(surface.get_height()/2)
            icon_size = int(self.radius * 0.7)
            points = []
            
            for i in range(5):
                # Outer points
                angle = i * 2 * math.pi / 5 - math.pi/2  # Start at top
                x = center_x + math.cos(angle) * icon_size
                y = center_y + math.sin(angle) * icon_size
                points.append((x, y))
                
                # Inner points
                angle += math.pi / 5
                x = center_x + math.cos(angle) * (icon_size * 0.4)
                y = center_y + math.sin(angle) * (icon_size * 0.4)
                points.append((x, y))
            
            pygame.draw.polygon(surface, (255, 255, 255), points)
        
        # Rotate and draw the power-up
        rotated_surface = pygame.transform.rotate(surface, self.rotation)
        new_rect = rotated_surface.get_rect(center=(self.x, self.y))
        screen.blit(rotated_surface, new_rect.topleft)

class ComboText:
    def __init__(self, x, y, combo, points):
        self.x = x
        self.y = y
        self.combo = combo
        self.points = points
        self.lifetime = 1.5  # Display for 1.5 seconds
        self.time_alive = 0
        self.vy = -80  # Float upward
        self.scale = 1.5  # Initial scale (for pop-in effect)
    
    def update(self, dt):
        self.time_alive += dt
        self.y += self.vy * dt
        
        # Scale animation
        if self.time_alive < 0.2:
            # Pop in
            self.scale = 1.5 - 0.5 * (self.time_alive / 0.2)
        elif self.time_alive > self.lifetime - 0.3:
            # Fade out
            self.scale = 1.0 - 0.5 * ((self.time_alive - (self.lifetime - 0.3)) / 0.3)
    
    def draw(self, screen, font):
        if self.time_alive < self.lifetime:
            # Calculate alpha for fade-out
            alpha = 255
            if self.time_alive > self.lifetime - 0.3:
                # Fade out in last 0.3 seconds
                fade_time = (self.time_alive - (self.lifetime - 0.3)) / 0.3
                alpha = int(255 * (1 - fade_time))
            
            # Create text
            combo_text = f"{self.combo}X COMBO!"
            points_text = f"+{self.points}"
            
            # Render combo text
            combo_surface = font.render(combo_text, True, (255, 200, 0))
            combo_surface.set_alpha(alpha)
            scaled_combo = pygame.transform.scale(combo_surface, 
                                               (int(combo_surface.get_width() * self.scale), 
                                                int(combo_surface.get_height() * self.scale)))
            combo_rect = scaled_combo.get_rect(center=(self.x, self.y))
            screen.blit(scaled_combo, combo_rect.topleft)
            
            # Render points text below
            points_surface = font.render(points_text, True, (255, 255, 255))
            points_surface.set_alpha(alpha)
            scaled_points = pygame.transform.scale(points_surface, 
                                               (int(points_surface.get_width() * self.scale * 0.8), 
                                                int(points_surface.get_height() * self.scale * 0.8)))
            points_rect = scaled_points.get_rect(center=(self.x, self.y + 30))
            screen.blit(scaled_points, points_rect.topleft)
    
    def is_expired(self):
        return self.time_alive >= self.lifetime

def spawn_fruit(screen_width, screen_height, multiplier):
    x = random.randint(50, screen_width - 50)
    y = screen_height + 50
    radius = random.randint(25, 35)
    vx = random.uniform(-100, 100) * multiplier
    vy = random.uniform(-600, -400) * multiplier
    return Fruit(x, y, radius, None, vx, vy)

def spawn_bomb(screen_width, screen_height, multiplier):
    x = random.randint(50, screen_width - 50)
    y = screen_height + 50
    radius = random.randint(25, 35)
    vx = random.uniform(-100, 100) * multiplier
    vy = random.uniform(-600, -400) * multiplier
    return Bomb(x, y, radius, (0, 0, 0), vx, vy)

def spawn_powerup(screen_width, screen_height, multiplier):
    x = random.randint(50, screen_width - 50)
    y = screen_height + 50
    radius = random.randint(20, 30)
    vx = random.uniform(-100, 100) * multiplier
    vy = random.uniform(-600, -400) * multiplier
    power_type = random.choice(["freeze", "bonus"])
    return PowerUp(x, y, radius, power_type, vx, vy)

# -------------------------------
# Main Game Function
# -------------------------------

def main():
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.7,
                          min_tracking_confidence=0.5)

    # Initialize webcam
    cap = cv2.VideoCapture(0)  # Change to 0 if needed.
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set webcam resolution
    screen_width, screen_height = 800, 600
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

    # Initialize Pygame
    pygame.init()
    pygame.mixer.init()  # Initialize audio system
    
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Enhanced Fruit Ninja with Hand Tracking")
    clock = pygame.time.Clock()

    # Load sounds
    try:
        slice_sound = pygame.mixer.Sound("sounds/slice.mp3")
        bomb_sound = pygame.mixer.Sound("sounds/bomb.mp3")
        combo_sound = pygame.mixer.Sound("sounds/combo.mp3")
        powerup_sound = pygame.mixer.Sound("sounds/powerup.mp3")
    except Exception as e:
        # Create default sounds if files not found
        print(f"Sound files not found: {e}")
        # Create empty sounds as fallback
        slice_sound = pygame.mixer.Sound(buffer=bytes([0, 0, 0, 0]))
        bomb_sound = pygame.mixer.Sound(buffer=bytes([0, 0, 0, 0]))
        combo_sound = pygame.mixer.Sound(buffer=bytes([0, 0, 0, 0]))
        powerup_sound = pygame.mixer.Sound(buffer=bytes([0, 0, 0, 0]))
    
    # Use system fonts instead of loading custom fonts
    main_font = pygame.font.SysFont(None, 36)
    combo_font = pygame.font.SysFont(None, 30)
    
    # Load background
    try:
        background = pygame.image.load("images/dojo_background.jpg")
        background = pygame.transform.scale(background, (screen_width, screen_height))
    except Exception as e:
        print(f"Background image not found: {e}")
        # Create gradient background
        background = pygame.Surface((screen_width, screen_height))
        for y in range(screen_height):
            # Create a blue to dark blue gradient
            color = (0, 0, 50 + int(150 * (1 - y / screen_height)))
            pygame.draw.line(background, color, (0, y), (screen_width, y))
    
    # Initialize game variables
    score = 0
    previous_score_milestone = 0  # Track the last milestone for glitter effects
    highscore = 0
    fruits = []
    slice_effects = []
    combo_texts = []
    glitter_effects = []  
    trail_points = [[], []]  # One trail for each hand (move this from the loop)
    spawn_timer = 0.0
    game_time = 0.0
    combo_counter = 0
    combo_timer = 0.0
    # lives = 3  # Remove this line
    
    # Power-up variables
    freeze_active = False
    freeze_timer = 0
    score_multiplier = 1
    multiplier_timer = 0
    
    # Game state variables
    game_state = "menu"  # "menu", "game", "game_over"
    game_over_timer = 0
    
    # Smoothing factor for hand centroid
    alpha = 0.3
    
    # Variables for hand tracking
    hand_centroid = None
    prev_hand_centroid = None
    game_pointer = None
    scaling_factor = 1.5
    
    # Trail for swipe
    max_trail_length = 5  # Decreased from 10 to 5 for shorter trail
    trail_colors = []  # Colors for trail gradient
    
    # Initialize trail colors (gradient from green to cyan)
    for i in range(max_trail_length):
        t = i / max_trail_length
        r = int(0 * (1-t) + 0 * t)
        g = int(255 * (1-t) + 100 * t)
        b = int(0 * (1-t) + 255 * t)
        a = int(255 * (1-t) + 100 * t)
        trail_colors.append((r, g, b, a))
    
    # Performance metrics
    frame_times = []
    max_frames_to_track = 60
    
    # Menu buttons
    play_button_rect = pygame.Rect(screen_width//2 - 100, screen_height//2 - 50, 200, 80)
    quit_button_rect = pygame.Rect(screen_width//2 - 100, screen_height//2 + 50, 200, 80)
    
    running = True
    while running:
        dt = min(clock.tick(60) / 1000.0, 0.05)  # Cap dt to prevent physics issues on lag
        frame_start_time = time.time()
        
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if game_state == "game":
                        game_state = "menu"
                    else:
                        running = False
                elif event.key == pygame.K_SPACE and game_state == "menu":
                    game_state = "game"
                    # Reset game variables
                    score = 0
                    fruits = []
                    slice_effects = []
                    combo_texts = []
                    spawn_timer = 0.0
                    game_time = 0.0
                    combo_counter = 0
                    combo_timer = 0.0
                    lives = 3
                    freeze_active = False
                    score_multiplier = 1
        
        # Process webcam
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Flip and convert frame
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        # Track both hands
        game_pointers = []  # List to store both hand positions
        
        if results.multi_hand_landmarks:
            h, w, _ = frame.shape
            
            # Process all detected hands (up to 2)
            for handLms in results.multi_hand_landmarks:
                # Get index finger tip position for precise pointing (landmark 8)
                index_finger_tip = handLms.landmark[8]
                pointer_x = int(index_finger_tip.x * w)
                pointer_y = int(index_finger_tip.y * h)
                
                # Map webcam coordinates to game screen coordinates
                game_pointer = (
                    int(pointer_x * screen_width / w),
                    int(pointer_y * screen_height / h)
                )
                game_pointers.append(game_pointer)
                
                # Draw landmarks on frame for hand detection window
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
        
        cv2.imshow("Hand Detection", frame)
        cv2.waitKey(1)

        # Update trails for both hands
        # First, handle removal of trails for hands that disappeared
        if len(game_pointers) < len(trail_points):
            # Remove trails for hands that are no longer detected
            trail_points = trail_points[:len(game_pointers)]
        
        # Then update trails for detected hands
        for i, pointer in enumerate(game_pointers):
            # Add new trail for new hand if needed
            if i >= len(trail_points):
                trail_points.append([])
            
            # Add current position to trail
            trail_points[i].append(pointer)
            
            # Limit trail length
            if len(trail_points[i]) > max_trail_length:
                trail_points[i].pop(0)
        
        # Determine slicing lines for all hands
        slicing_lines = []
        for hand_trail in trail_points:
            if len(hand_trail) >= 2:
                slicing_lines.append((hand_trail[-2], hand_trail[-1]))
        
        # Draw background
        screen.blit(background, (0, 0))
        
        # Menu state
        if game_state == "menu":
            # Draw title
            title_text = main_font.render("ENHANCED FRUIT NINJA", True, (255, 255, 255))
            title_rect = title_text.get_rect(center=(screen_width//2, screen_height//4))
            screen.blit(title_text, title_rect)
            
            # Draw buttons
            pygame.draw.rect(screen, (50, 150, 50), play_button_rect, border_radius=10)
            pygame.draw.rect(screen, (150, 50, 50), quit_button_rect, border_radius=10)
            
            play_text = main_font.render("PLAY", True, (255, 255, 255))
            play_text_rect = play_text.get_rect(center=play_button_rect.center)
            screen.blit(play_text, play_text_rect)
            
            quit_text = main_font.render("QUIT", True, (255, 255, 255))
            quit_text_rect = quit_text.get_rect(center=quit_button_rect.center)
            screen.blit(quit_text, quit_text_rect)
            
            # Handle button clicks
            if game_pointer and pygame.mouse.get_pressed()[0]:
                if play_button_rect.collidepoint(game_pointer):
                    game_state = "game"
                    # Reset game variables
                    score = 0
                    fruits = []
                    slice_effects = []
                    combo_texts = []
                    spawn_timer = 0.0
                    game_time = 0.0
                    combo_counter = 0
                    combo_timer = 0.0
                    lives = 3
                    freeze_active = False
                    score_multiplier = 1
                elif quit_button_rect.collidepoint(game_pointer):
                    running = False
        
        # Game state
        elif game_state == "game":
            game_time += dt
            
            # Update power-up timers
            if freeze_active:
                freeze_timer -= dt
                if freeze_timer <= 0:
                    freeze_active = False
            
            if score_multiplier > 1:
                multiplier_timer -= dt
                if multiplier_timer <= 0:
                    score_multiplier = 1
            
            # Spawn objects
            if not freeze_active:
                spawn_timer += dt
                spawn_interval = max(0.8, 2.0 - (game_time / 60.0))  # Gradually decrease spawn interval
                
                if spawn_timer > spawn_interval:
                    spawn_timer = 0.0
                    
                    # Determine what to spawn based on probabilities
                    spawn_type = random.random()
                    # Speed multiplier increases with time
                    speed_multiplier = 1 + (game_time / 120.0)
                    speed_multiplier = min(speed_multiplier, 2.0)
                    
                    if spawn_type < 0.05:  # 5% chance for power-up
                        fruits.append(spawn_powerup(screen_width, screen_height, speed_multiplier))
                    elif spawn_type < 0.20:  # 15% chance for bomb
                        fruits.append(spawn_bomb(screen_width, screen_height, speed_multiplier))
                    else:  # 80% chance for fruit
                        # Spawn 1-3 fruits
                        fruit_count = random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
                        for _ in range(fruit_count):
                            fruits.append(spawn_fruit(screen_width, screen_height, speed_multiplier))
            
            # Update objects
            for obj in fruits:
                obj.update(dt)
            
            # Check for missed fruits (not bombs)
            for obj in fruits[:]:
                if obj.y - obj.radius > screen_height and not obj.sliced and not obj.is_bomb and not isinstance(obj, PowerUp):
                    fruits.remove(obj)
            
            # Remove objects that are off-screen
            fruits = [obj for obj in fruits if obj.y - obj.radius < screen_height + 100]
            
            # Update combo timer
            if combo_counter > 0:
                combo_timer -= dt
                if combo_timer <= 0:
                    combo_counter = 0
            
            # Check for slicing with any hand
            newly_sliced = []
            for slicing_line in slicing_lines:
                A, B = slicing_line
                for obj in fruits:
                    if not obj.sliced and line_circle_collision(A, B, (obj.x, obj.y), obj.radius):
                        obj.sliced = True
                        newly_sliced.append(obj)
                        
                        # Create slice effect
                        slice_effects.append(SliceEffect(obj.x, obj.y, obj.color if not isinstance(obj, PowerUp) else (255, 255, 255)))
                        
                        if isinstance(obj, PowerUp):
                            if obj.power_type == "freeze":
                                freeze_active = True
                                freeze_timer = 5.0  # 5 seconds freeze
                                pygame.mixer.Sound.play(powerup_sound)
                            elif obj.power_type == "bonus":
                                score_multiplier = 2
                                multiplier_timer = 10.0  # 10 seconds of double points
                                pygame.mixer.Sound.play(powerup_sound)
                        elif obj.is_bomb:
                            # Sliced a bomb - game over
                            pygame.mixer.Sound.play(bomb_sound)
                            # Flash screen red
                            flash_surface = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
                            flash_surface.fill((255, 0, 0, 128))
                            screen.blit(flash_surface, (0, 0))
                            # Game over when bomb is hit
                            game_state = "game_over"
                            if score > highscore:
                                highscore = score
                        else:
                            # Sliced a fruit
                            pygame.mixer.Sound.play(slice_sound)
                            points = obj.points * score_multiplier
                            score += points
                            
                            # Check for 10-point milestone for glitter effect
                            if (score // 10) > (previous_score_milestone // 10):
                                # Trigger glitter effect at the center of the screen
                                glitter_effects.append(GlitterEffect(screen_width // 2, screen_height // 2))
                                previous_score_milestone = score
                            
                            # Update combo
                            combo_counter += 1
            
            # Update slice effects
            for effect in slice_effects[:]:
                effect.update(dt)
                if effect.is_expired():
                    slice_effects.remove(effect)
            
            # Update combo texts
            for text in combo_texts[:]:
                text.update(dt)
                if text.is_expired():
                    combo_texts.remove(text)
            
            # Update glitter effects
            for effect in glitter_effects[:]:
                effect.update(dt)
                if effect.is_expired():
                    glitter_effects.remove(effect)
            
            # Draw all game objects
            for obj in sorted(fruits, key=lambda x: x.y):  # Sort by y to draw correct depth
                obj.draw(screen)
            
            # Draw slice effects
            for effect in slice_effects:
                effect.draw(screen)
                
            # Draw glitter effects
            for effect in glitter_effects:
                effect.draw(screen)
            
            # Draw trail for each hand
            for i, hand_trail in enumerate(trail_points):
                if len(hand_trail) > 1:
                    # Draw gradient trail
                    for j in range(len(hand_trail) - 1):
                        start = hand_trail[j]
                        end = hand_trail[j + 1]
                        color = trail_colors[j]
                        pygame.draw.line(screen, color, start, end, 3)
                        # Add glow
                        for width in range(5, 1, -1):
                            glow_color = (*color[:3], color[3] // (6-width))
                            pygame.draw.line(screen, glow_color, start, end, width*2)
            
            # Draw game pointers
            for pointer in game_pointers:
                # Draw a circular cursor with glow
                for radius in range(15, 5, -3):
                    alpha = 100 if radius == 15 else 200
                    glow_color = (255, 255, 255, alpha)
                    gfxdraw.aacircle(screen, pointer[0], pointer[1], radius, glow_color)
                
                # Add a more visible dot/blade tip marker
                blade_tip_color = (255, 255, 255)  # Bright white for visibility
                pygame.draw.circle(screen, blade_tip_color, pointer, 3)  # Small solid circle
                
                # Draw a highlight in the center for better visibility
                highlight_color = (200, 255, 200)  # Light green highlight
                pygame.draw.circle(screen, highlight_color, pointer, 1)
            
            # Draw UI
            ui_panel = pygame.Surface((screen_width, 50), pygame.SRCALPHA)
            ui_panel.fill((0, 0, 0, 150))
            screen.blit(ui_panel, (0, 0))
            
            # Draw score
            score_text = main_font.render(f"Score: {score}", True, (255, 255, 255))
            screen.blit(score_text, (20, 10))
            
            # Draw combo counter if active
            if combo_counter > 1:
                combo_text = main_font.render(f"Combo: {combo_counter}x", True, (255, 200, 0))
                screen.blit(combo_text, (screen_width // 2 - combo_text.get_width() // 2, 10))
            
            # Draw active power-ups
            if freeze_active:
                freeze_text = main_font.render(f"FREEZE: {int(freeze_timer)}", True, (120, 200, 255))
                screen.blit(freeze_text, (screen_width - 200, 10))
            
            if score_multiplier > 1:
                multi_text = main_font.render(f"2X POINTS: {int(multiplier_timer)}", True, (255, 215, 0))
                screen.blit(multi_text, (screen_width - 200, 40))
        
        # Game over state
        elif game_state == "game_over":
            game_over_timer += dt
            
            # Draw semi-transparent overlay
            overlay = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            screen.blit(overlay, (0, 0))
            
            # Draw game over text
            game_over_text = main_font.render("GAME OVER", True, (255, 0, 0))
            game_over_rect = game_over_text.get_rect(center=(screen_width//2, screen_height//3))
            screen.blit(game_over_text, game_over_rect)
            
            # Draw final score
            score_text = main_font.render(f"Score: {score}", True, (255, 255, 255))
            score_rect = score_text.get_rect(center=(screen_width//2, screen_height//3 + 60))
            screen.blit(score_text, score_rect)
            
            # Draw high score
            high_score_text = main_font.render(f"High Score: {highscore}", True, (255, 215, 0))
            high_score_rect = high_score_text.get_rect(center=(screen_width//2, screen_height//3 + 100))
            screen.blit(high_score_text, high_score_rect)
            
            # Draw restart button
            restart_button_rect = pygame.Rect(screen_width//2 - 100, screen_height//2 + 50, 200, 60)
            pygame.draw.rect(screen, (50, 150, 50), restart_button_rect, border_radius=10)
            restart_text = main_font.render("RESTART", True, (255, 255, 255))
            restart_text_rect = restart_text.get_rect(center=restart_button_rect.center)
            screen.blit(restart_text, restart_text_rect)
            
            # Draw menu button
            menu_button_rect = pygame.Rect(screen_width//2 - 100, screen_height//2 + 130, 200, 60)
            pygame.draw.rect(screen, (100, 100, 150), menu_button_rect, border_radius=10)
            menu_text = main_font.render("MENU", True, (255, 255, 255))
            menu_text_rect = menu_text.get_rect(center=menu_button_rect.center)
            screen.blit(menu_text, menu_text_rect)
            
            # Handle button clicks after a delay
            if game_over_timer > 1.0 and game_pointer and pygame.mouse.get_pressed()[0]:
                if restart_button_rect.collidepoint(game_pointer):
                    game_state = "game"
                    # Reset game variables
                    score = 0
                    fruits = []
                    slice_effects = []
                    combo_texts = []
                    spawn_timer = 0.0
                    game_time = 0.0
                    combo_counter = 0
                    combo_timer = 0.0
                    lives = 3
                    freeze_active = False
                    score_multiplier = 1
                elif menu_button_rect.collidepoint(game_pointer):
                    game_state = "menu"
        
        # Display FPS in the corner
        frame_end_time = time.time()
        frame_time = frame_end_time - frame_start_time
        frame_times.append(frame_time)
        if len(frame_times) > max_frames_to_track:
            frame_times.pop(0)
        avg_frame_time = sum(frame_times) / len(frame_times)
        fps = 1 / avg_frame_time if avg_frame_time > 0 else 0
        fps_text = main_font.render(f"FPS: {int(fps)}", True, (255, 255, 255))
        screen.blit(fps_text, (screen_width - 100, screen_height - 30))
        
        # Update display
        pygame.display.flip()
    
    # Clean up
    cap.release()
    pygame.quit()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()