import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

import cv2
import mediapipe as mp
import math
import numpy as np
from collections import deque

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
    max_num_hands=2
)
mp_drawing = mp.solutions.drawing_utils

def calculate_distance(p1, p2):
    
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def is_thumb_up(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    thumb_up = thumb_tip.y < thumb_ip.y < index_mcp.y
    curled = all(
        landmarks[tip].y > landmarks[mcp].y
        for tip, mcp in [
            (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP),
            (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
            (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_MCP),
            (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_MCP),
        ]
    )
    return thumb_up and curled

def is_thumb_down(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    thumb_down = thumb_tip.y > thumb_ip.y > index_mcp.y
    curled = all(
        landmarks[tip].y > landmarks[mcp].y
        for tip, mcp in [
            (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP),
            (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
            (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_MCP),
            (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_MCP),
        ]
    )
    return thumb_down and curled

def is_open_hand(landmarks):
    return all(
        landmarks[tip].y < landmarks[mcp].y
        for tip, mcp in [
            (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP),
            (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
            (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_MCP),
            (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_MCP),
        ]
    )

def is_closed_fist(landmarks):
    return all(
        landmarks[tip].y > landmarks[mcp].y
        for tip, mcp in [
            (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP),
            (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
            (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_MCP),
            (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_MCP),
        ]
    )

def detect_heart_gesture(all_hands):
    if len(all_hands) < 2:
        return False
    thumb1 = all_hands[0].landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb2 = all_hands[1].landmark[mp_hands.HandLandmark.THUMB_TIP]
    index1 = all_hands[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index2 = all_hands[1].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    close_thumb = calculate_distance(thumb1, thumb2) < 0.15
    close_index = calculate_distance(index1, index2) < 0.15
    return close_thumb and close_index

def is_double_thumb_up(all_hands):
    if len(all_hands) < 2:
        return False
    return all(is_thumb_up(hand.landmark) for hand in all_hands)

def is_two_fingers_straight(landmarks):
    index_straight = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
    middle_straight = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
    ring_curled = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y
    pinky_curled = landmarks[mp_hands.HandLandmark.PINKY_TIP].y > landmarks[mp_hands.HandLandmark.PINKY_MCP].y
    return index_straight and middle_straight and ring_curled and pinky_curled

def get_stable_gesture(gesture_queue, current_gesture, stability=5):
    if not current_gesture:
        return None
    gesture_queue.append(current_gesture)
    if len(gesture_queue) > stability:
        gesture_queue.popleft()
    if gesture_queue.count(current_gesture) > stability // 2:
        return current_gesture
    return None

def gesture_recognition():
    cap = cv2.VideoCapture(0)
    print("Press 'S' to start camera, 'Q' to quit.")
    started = False
    gesture_queue = deque(maxlen=10)
    stable_gesture = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading webcam.")
            break

        white_bg = 255 * np.ones_like(frame, dtype=np.uint8)
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        gesture_detected = ""

        if started and results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(white_bg, hand, mp_hands.HAND_CONNECTIONS)

            if detect_heart_gesture(results.multi_hand_landmarks):
                gesture_detected = "LOVE"
            elif is_double_thumb_up(results.multi_hand_landmarks):
                gesture_detected = "THANK YOU"
            else:
                for hand in results.multi_hand_landmarks:
                    lms = hand.landmark
                    if is_thumb_up(lms):
                        gesture_detected = "OK"
                        break
                    elif is_thumb_down(lms):
                        gesture_detected = "DISAPPROVAL"
                        break
                    elif is_open_hand(lms):
                        gesture_detected = "HELLO"
                        break
                    elif is_closed_fist(lms):
                        gesture_detected = "STOP"
                        break
                    elif is_two_fingers_straight(lms):
                        gesture_detected = "SORRY"
                        break

            stable = get_stable_gesture(gesture_queue, gesture_detected)
            if stable:
                stable_gesture = stable

        if not started:
            cv2.putText(white_bg, "Press 'S' to Start", (100, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        elif stable_gesture:
            cv2.putText(white_bg, stable_gesture, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 150, 0), 3)

        cv2.imshow("Gesture Recognition", white_bg)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            started = True
            print("Camera started.")
        elif key == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed.")

if __name__ == "__main__":
    gesture_recognition()
