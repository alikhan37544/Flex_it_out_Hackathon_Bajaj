import numpy as np
import collections

class MotionDetector:
    def __init__(self, velocity_threshold=0.05, acceleration_threshold=0.02, buffer_size=5):
        """
        velocity_threshold: Minimum frame-to-frame change (in normalized coordinates) to consider as fast motion.
        acceleration_threshold: Minimum change in velocity between frames to consider as a swing.
        buffer_size: Number of frames to consider for detecting oscillatory (waving) motion.
        """
        self.prev_position = None
        self.prev_velocity = None
        self.velocity_threshold = velocity_threshold
        self.acceleration_threshold = acceleration_threshold
        self.buffer_size = buffer_size
        self.x_buffer = collections.deque(maxlen=buffer_size)

    def update(self, position):
        """
        position: Tuple of (x, y, z) in normalized coordinates.
        Returns a string gesture if detected, or None.
        """
        current_position = np.array(position)
        gesture = None

        # Update oscillation buffer for horizontal (waving) detection.
        self.x_buffer.append(current_position[0])
        
        if self.prev_position is None:
            # First frame: initialize state.
            self.prev_position = current_position
            self.prev_velocity = np.array([0, 0, 0])
            return None
        
        # Calculate velocity and acceleration (frame-to-frame differences).
        velocity = current_position - self.prev_position
        acceleration = velocity - self.prev_velocity

        # Update stored values for next iteration.
        self.prev_position = current_position
        self.prev_velocity = velocity

        # --- Detect Waving (oscillation in X) ---
        if len(self.x_buffer) == self.buffer_size:
            min_x = min(self.x_buffer)
            max_x = max(self.x_buffer)
            # If the horizontal range exceeds the threshold, we may have a wave gesture.
            if max_x - min_x > self.velocity_threshold:
                gesture = "Waving Hi"

        # --- Detect Tennis Swing (fast forward/backward motion) ---
        # We consider it a swing if the overall velocity and acceleration are high,
        # and the dominant change is in the z-axis (depth).
        if np.linalg.norm(velocity) > self.velocity_threshold and np.linalg.norm(acceleration) > self.acceleration_threshold:
            # Check if z-axis movement is the strongest component.
            if abs(velocity[2]) > abs(velocity[0]) and abs(velocity[2]) > abs(velocity[1]):
                gesture = "Tennis Swing"

        return gesture
