def is_peace_sign(hand_landmarks):
    lm = hand_landmarks.landmark
    # Check if index and middle fingers are extended.
    index_extended = lm[8].y < lm[6].y
    middle_extended = lm[12].y < lm[10].y
    # Check if ring and pinky fingers are folded.
    ring_folded = lm[16].y > lm[14].y
    pinky_folded = lm[20].y > lm[18].y
    return index_extended and middle_extended and ring_folded and pinky_folded
