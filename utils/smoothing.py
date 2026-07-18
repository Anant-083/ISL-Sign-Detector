"""
utils/smoothing.py

Prevents flickering/false-trigger predictions by requiring
a majority vote across the last N predictions before accepting a sign.
"""

from collections import deque, Counter


class PredictionSmoother:
    def __init__(self, window_size=10, confidence_threshold=0.80, majority_ratio=0.6):
        """
        window_size: how many recent predictions to remember
        confidence_threshold: minimum model confidence to even consider a prediction
        majority_ratio: fraction of the window that must agree to accept a sign
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.majority_ratio = majority_ratio
        self.history = deque(maxlen=window_size)
        self.last_confirmed = None

    def update(self, predicted_label, confidence):
        """
        Call this every frame/prediction cycle.
        Returns a confirmed label (str) if a stable sign is detected, else None.
        """
        if confidence < self.confidence_threshold:
            # Low confidence prediction — still add "None" as a placeholder so
            # a shaky/uncertain moment doesn't get swallowed silently
            self.history.append(None)
            return None

        self.history.append(predicted_label)

        if len(self.history) < self.window_size:
            return None  # not enough data yet to decide

        counts = Counter(self.history)
        most_common_label, count = counts.most_common(1)[0]

        if most_common_label is None:
            return None

        if count / self.window_size >= self.majority_ratio:
            # Avoid re-confirming the same sign repeatedly every frame
            if most_common_label != self.last_confirmed:
                self.last_confirmed = most_common_label
                return most_common_label

        return None

    def reset(self):
        self.history.clear()
        self.last_confirmed = None


class WordBuffer:
    """
    Accumulates confirmed signs into a running sentence buffer.
    """

    def __init__(self, max_words=20):
        self.words = []
        self.max_words = max_words

    def add_word(self, word):
        if word is None:
            return
        self.words.append(word)
        if len(self.words) > self.max_words:
            self.words.pop(0)

    def get_sentence_raw(self):
        return " ".join(self.words)

    def clear(self):
        self.words = []