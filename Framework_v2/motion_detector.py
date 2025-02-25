import numpy as np
import collections

class MotionDetector:
    def __init__(self, velocity_threshold=0.05, acceleration_threshold=0.02, buffer_size=5):
        """
        Initializes the MotionDetector with specified thresholds and buffer size.
        
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
        Updates the detector with the current position and returns a detected gesture if any.
        
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

        # Detect gestures
        gesture = self.detect_waving()
        if not gesture:
            gesture = self.detect_tennis_swing(velocity, acceleration)
        if not gesture:
            gesture = self.detect_jump(velocity)
        if not gesture:
            gesture = self.detect_duck(velocity)
        if not gesture:
            gesture = self.detect_punch(velocity)

        return gesture

    def detect_waving(self):
        """
        Detects waving gesture based on oscillation in the x-axis.
        """
        if len(self.x_buffer) == self.buffer_size:
            min_x = min(self.x_buffer)
            max_x = max(self.x_buffer)
            if max_x - min_x > self.velocity_threshold:
                return "Waving Hi"
        return None

    def detect_tennis_swing(self, velocity, acceleration):
        """
        Detects tennis swing gesture based on fast forward/backward motion.
        """
        if np.linalg.norm(velocity) > self.velocity_threshold and np.linalg.norm(acceleration) > self.acceleration_threshold:
            if abs(velocity[2]) > abs(velocity[0]) and abs(velocity[2]) > abs(velocity[1]):
                return "Tennis Swing"
        return None

    def detect_jump(self, velocity):
        """
        Detects jumping gesture based on upward motion.
        """
        if velocity[1] < -self.velocity_threshold:
            return "Jump"
        return None

    def detect_duck(self, velocity):
        """
        Detects ducking gesture based on downward motion.
        """
        if velocity[1] > self.velocity_threshold:
            return "Duck"
        return None

    def detect_punch(self, velocity):
        """
        Detects punching gesture based on forward motion.
        """
        if velocity[2] < -self.velocity_threshold:
            return "Punch"
        return None

    def reset(self):
        """
        Resets the detector state.
        """
        self.prev_position = None
        self.prev_velocity = None
        self.x_buffer.clear()