import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


class SignLanguageDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect_landmarks(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on frame
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

        return frame, landmarks

    def preprocess_landmarks(self, landmarks):
        if not landmarks:
            return np.zeros((21, 3))  # Return zeros if no hand detected
        landmarks = np.array(landmarks)
        wrist = landmarks[0]
        landmarks = landmarks - wrist

        max_val = np.max(np.abs(landmarks))
        if max_val != 0:
            landmarks = landmarks / max_val

        return landmarks


def main():
    cap = cv2.VideoCapture(0)
    detector = SignLanguageDetector()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        # Detect hand landmarks
        frame, landmarks = detector.detect_landmarks(frame)

        if landmarks:
            processed_landmarks = detector.preprocess_landmarks(landmarks)

        cv2.imshow('Sign Language Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()