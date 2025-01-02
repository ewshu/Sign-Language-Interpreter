import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import tensorflow as tf
from sign_model import SignModel


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

        self.sequence_length = 30
        self.sequence = deque(maxlen=self.sequence_length)

        self.model = SignModel(num_classes=10)

        self.signs = {
            0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four",
            5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"
        }

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

    def collect_data(self, sign_label, num_sequences=30):
        """Collect training data for a specific sign"""
        sequences = []
        cap = cv2.VideoCapture(0)
        sequence_count = 0

        print(f"\nCollecting data for sign {self.signs[sign_label]}")
        print("Press 'r' to record a sequence")
        print("Press 'q' to finish collecting for this sign")

        while sequence_count < num_sequences:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            frame, _ = self.detect_landmarks(frame)

            # Add status text
            cv2.putText(frame, f"Sign: {self.signs[sign_label]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Sequences recorded: {sequence_count}/{num_sequences}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'r' to record, 'q' to finish", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Collecting Data', frame)
            key = cv2.waitKey(1)

            if key & 0xFF == ord('q'):
                break

            if key & 0xFF == ord('r'):
                print(f"Recording sequence {sequence_count + 1}")
                current_sequence = []

                # Countdown
                for countdown in range(3, 0, -1):
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    frame = cv2.flip(frame, 1)
                    cv2.putText(frame, str(countdown), (200, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 0, 0), 4)
                    cv2.imshow('Collecting Data', frame)
                    cv2.waitKey(1000)

                for _ in range(self.sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    frame = cv2.flip(frame, 1)
                    frame, landmarks = self.detect_landmarks(frame)

                    if landmarks:
                        processed_landmarks = self.preprocess_landmarks(landmarks)
                        current_sequence.append(processed_landmarks.flatten())

                    cv2.putText(frame, "Recording...", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Collecting Data', frame)
                    cv2.waitKey(1)

                if len(current_sequence) == self.sequence_length:
                    sequences.append(current_sequence)
                    sequence_count += 1
                    print(f"Sequence {sequence_count} recorded")

        cap.release()
        cv2.destroyAllWindows()
        return np.array(sequences)

    def train_model(self, X, y):
        """Train the model with collected data"""
        self.model.train(np.array(X), np.array(y))

    def predict_sign(self):
        """Predict the current sign from the sequence"""
        if len(self.sequence) < self.sequence_length:
            return None

        sequence_array = np.array(list(self.sequence))
        prediction = self.model.predict(sequence_array)
        predicted_sign = self.signs[np.argmax(prediction)]
        confidence = np.max(prediction)

        return predicted_sign, confidence


def main():
    detector = SignLanguageDetector()

    # Collect training data
    print("Starting data collection...")
    X = []
    y = []

    try:
        for sign in range(10):  # Collect data for numbers 0-9
            sequences = detector.collect_data(sign)
            if len(sequences) > 0:  # Only add if we got some sequences
                X.extend(sequences)
                y.extend([sign] * len(sequences))

            print(f"Completed collecting data for sign {sign}")
            proceed = input("Press Enter to continue to next sign, or 'q' to finish collection: ")
            if proceed.lower() == 'q':
                break

        if len(X) > 0:  # Only train if we have data
            X = np.array(X)
            y = tf.keras.utils.to_categorical(y)

            # Train the model
            print("Training model...")
            detector.train_model(X, y)

            print("Starting real-time detection...")
            cap = cv2.VideoCapture(0)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                frame, landmarks = detector.detect_landmarks(frame)

                if landmarks:
                    processed_landmarks = detector.preprocess_landmarks(landmarks)
                    detector.sequence.append(processed_landmarks.flatten())

                    prediction = detector.predict_sign()
                    if prediction:
                        sign, confidence = prediction
                        cv2.putText(frame, f"Sign: {sign} ({confidence:.2f})",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow('Sign Language Detector', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()