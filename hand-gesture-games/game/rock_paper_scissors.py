import cv2
import mediapipe as mp
import numpy as np
import random
import time

from motion_detector import MotionDetector
from thumbs_up import is_thumbs_up
from peace_sign import is_peace_sign
from closed_fist import is_closed_fist
from open_fist import is_open_fist

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)

# Initialize the motion detector (not used in this game but available if needed).
motion_detector = MotionDetector(velocity_threshold=0.05, acceleration_threshold=0.02, buffer_size=5)

cap = cv2.VideoCapture(0)

def decide_winner(user_choice, computer_choice):
    if user_choice == computer_choice:
        return "Tie"
    elif (user_choice == "rock" and computer_choice == "scissors") or \
         (user_choice == "paper" and computer_choice == "rock") or \
         (user_choice == "scissors" and computer_choice == "paper"):
        return "You win!"
    else:
        return "You lose!"

# Mapping gestures to game choices.
def gesture_to_choice(gesture):
    if gesture == "Thumbs Up":
        return "rock"
    elif gesture == "Open Fist":
        return "paper"
    elif gesture == "Peace Sign":
        return "scissors"
    else:
        return None

round_in_progress = False
result_text = ""
last_round_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror view.
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_height, frame_width, _ = frame.shape

    results = hands.process(frame_rgb)
    user_choice = None
    gesture_text = "No Gesture Detected"

    if results.multi_hand_landmarks:
        # Use the first found hand.
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Determine gesture by checking conditions.
            if is_thumbs_up(hand_landmarks) and not is_closed_fist(hand_landmarks):
                gesture_text = "Thumbs Up"
            elif is_peace_sign(hand_landmarks):
                gesture_text = "Peace Sign"
            elif is_closed_fist(hand_landmarks):
                gesture_text = "Closed Fist"
            elif is_open_fist(hand_landmarks):
                gesture_text = "Open Fist"
            else:
                gesture_text = "Unknown Gesture"

            # Map the detected gesture to a game choice.
            user_choice = gesture_to_choice(gesture_text)
            break  # Process only one hand

    # Display the detected gesture and (if available) the game choice.
    cv2.putText(frame, f"Gesture: {gesture_text}", (10, frame_height - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if user_choice and not round_in_progress:
        round_in_progress = True
        computer_choice = random.choice(["rock", "paper", "scissors"])
        outcome = decide_winner(user_choice, computer_choice)
        result_text = f"You: {user_choice} | Computer: {computer_choice} => {outcome}"
        last_round_time = time.time()

    # If a round is in progress, show the result on the frame.
    if round_in_progress:
        cv2.putText(frame, result_text, (10, frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # After 2 seconds, reset round.
        if time.time() - last_round_time > 2:
            round_in_progress = False
            result_text = ""

    cv2.imshow("Rock Paper Scissors - Gesture Game", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Escape key to exit.
        break

cap.release()
cv2.destroyAllWindows()