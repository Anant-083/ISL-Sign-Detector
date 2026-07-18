"""
utils/preprocess.py

Handles:
1. Extracting hand landmarks from a video frame using MediaPipe
2. Normalizing landmarks (wrist-relative, scale-invariant)
3. Building sequences for dynamic (LSTM) signs

Usage:
    from utils.preprocess import HandLandmarkExtractor

    extractor = HandLandmarkExtractor()
    landmarks = extractor.extract_from_frame(frame)   # single frame -> normalized landmarks
"""

import cv2
import mediapipe as mp
import numpy as np


class HandLandmarkExtractor:
    def __init__(self, max_hands=2, detection_confidence=0.6, tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def extract_from_frame(self, frame):
        """
        Takes a BGR frame (from OpenCV), returns normalized landmark array.
        Returns None if no hand detected.

        Output shape: (21, 3) per hand -> flattened to (63,) for one hand
        If two hands are needed, this returns a list of up to 2 hand arrays.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return None

        all_hands = []
        for hand_landmarks in results.multi_hand_landmarks:
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                dtype=np.float32,
            )  # shape (21, 3)

            normalized = self._normalize(coords)
            all_hands.append(normalized)

        return all_hands  # list of (21,3) arrays, one per detected hand

    def _normalize(self, coords):
        """
        Wrist-relative + scale-invariant normalization.
        coords: (21, 3) raw landmark array (landmark 0 = wrist)
        """
        wrist = coords[0].copy()
        shifted = coords - wrist  # translate so wrist = origin

        # Scale by the distance from wrist to middle finger MCP (landmark 9)
        # This makes the representation roughly invariant to hand size / distance from camera
        scale_ref = np.linalg.norm(shifted[9])
        if scale_ref < 1e-6:
            scale_ref = 1e-6  # avoid divide-by-zero

        normalized = shifted / scale_ref
        return normalized

    def draw_landmarks(self, frame, results):
        """Optional: draw landmarks on frame for debugging/visualization."""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        return frame

    def close(self):
        self.hands.close()


class SequenceBuffer:
    """
    Maintains a rolling buffer of frames for dynamic (LSTM) sign detection.
    """

    def __init__(self, sequence_length=30, feature_dim=63):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.buffer = []

    def add_frame(self, landmarks):
        """
        landmarks: flattened (feature_dim,) array for one frame.
        If no hand detected, pass a zero array of the same shape (padding).
        """
        self.buffer.append(landmarks)
        if len(self.buffer) > self.sequence_length:
            self.buffer.pop(0)

    def is_ready(self):
        return len(self.buffer) == self.sequence_length

    def get_sequence(self):
        """Returns (sequence_length, feature_dim) array, ready for LSTM input."""
        return np.array(self.buffer, dtype=np.float32)

    def reset(self):
        self.buffer = []


def flatten_hand(hand_array):
    """Converts (21,3) landmark array into a flat (63,) vector."""
    return hand_array.flatten()


def extract_landmarks_from_video(video_path, extractor, sequence_length=30):
    """
    Extracts a fixed-length sequence of hand landmarks from a video file.
    Used during dataset preprocessing (offline), not live inference.

    Returns: (sequence_length, 63) array, or None if no hand ever detected.
    """
    cap = cv2.VideoCapture(video_path)
    frames_landmarks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hands = extractor.extract_from_frame(frame)
        if hands:
            flat = flatten_hand(hands[0])  # use first detected hand
        else:
            flat = np.zeros(63, dtype=np.float32)  # padding for missed detection

        frames_landmarks.append(flat)

    cap.release()

    if len(frames_landmarks) == 0:
        return None

    frames_landmarks = np.array(frames_landmarks, dtype=np.float32)

    # Resample to fixed sequence_length using linear interpolation over frame indices
    if len(frames_landmarks) != sequence_length:
        indices = np.linspace(0, len(frames_landmarks) - 1, sequence_length).astype(int)
        frames_landmarks = frames_landmarks[indices]

    return frames_landmarks