import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import datetime


class SignModel:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.sequence_length = 30  # Number of frames to consider
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            LSTM(64, return_sequences=True, activation='relu', input_shape=(self.sequence_length, 63)),
            Dropout(0.2),
            LSTM(128, return_sequences=True, activation='relu'),
            Dropout(0.2),
            LSTM(64, return_sequences=False, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='Adam',
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy']
        )
        return model

    def train(self, X, y, epochs=50, batch_size=32):
        # Setup TensorBoard logging
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        return self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[tensorboard_callback]
        )

    def predict(self, sequence):
        return self.model.predict(np.expand_dims(sequence, axis=0))[0]