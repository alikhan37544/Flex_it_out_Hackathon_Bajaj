def is_closed_fist(hand_landmarks):
    lm = hand_landmarks.landmark
    index_folded = lm[8].y > lm[6].y
    middle_folded = lm[12].y > lm[10].y
    ring_folded = lm[16].y > lm[14].y
    pinky_folded = lm[20].y > lm[18].y
    return index_folded and middle_folded and ring_folded and pinky_folded