def is_open_fist(hand_landmarks):
    lm = hand_landmarks.landmark
    index_open = lm[8].y < lm[6].y
    middle_open = lm[12].y < lm[10].y
    ring_open = lm[16].y < lm[14].y
    pinky_open = lm[20].y < lm[18].y
    return index_open and middle_open and ring_open and pinky_open