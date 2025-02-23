import numpy as np

def is_thumbs_up(hand_landmarks):
    lm = hand_landmarks.landmark
    wrist = lm[0]
    thumb_tip = lm[4]
    thumb_ip = lm[3]
    index_mcp = lm[5]
    
    # Compute a rough hand size using the distance from the wrist to the middle fingertip.
    hand_size = np.linalg.norm(np.array([lm[0].x, lm[0].y]) - np.array([lm[12].x, lm[12].y]))
    thumb_distance = np.linalg.norm(np.array([wrist.x, wrist.y]) - np.array([thumb_tip.x, thumb_tip.y]))
    
    # Check if the thumb is extended upward:
    # - The thumb tip must be above both the wrist and thumb IP joint.
    # - Its distance from the wrist must be significant relative to hand size.
    thumb_extended = (thumb_tip.y < wrist.y and thumb_tip.y < thumb_ip.y and thumb_distance > 0.5 * hand_size)
    
    # Ensure that the thumb is sufficiently separated from the index finger.
    thumb_to_index_distance = np.linalg.norm(
        np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_mcp.x, index_mcp.y])
    )
    thumb_separated = thumb_to_index_distance > 0.3 * hand_size
    
    # Check that other fingers are folded.
    index_folded = lm[8].y > lm[6].y
    middle_folded = lm[12].y > lm[10].y
    ring_folded = lm[16].y > lm[14].y
    pinky_folded = lm[20].y > lm[18].y
    
    return thumb_extended and thumb_separated and index_folded and middle_folded and ring_folded and pinky_folded
