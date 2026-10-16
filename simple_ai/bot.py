"""A small intent-classifying chatbot built on multinomial Naive Bayes."""

import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_INTENTS = Path(__file__).with_name("intents.json")

FALLBACK_RESPONSES = [
    "Sorry, I didn't understand that. Could you rephrase?",
    "I'm not sure what you mean. Try asking what I can do.",
]


def tokenize(text):
    """Lowercase the text and split it into word tokens."""
    return re.findall(r"[a-z0-9']+", text.lower())


class NaiveBayesClassifier:
    """Multinomial Naive Bayes over bag-of-words features with Laplace smoothing."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_counts = Counter()
        self.word_counts = defaultdict(Counter)
        self.vocabulary = set()

    def train(self, examples):
        """Train on an iterable of (text, label) pairs."""
        for text, label in examples:
            tokens = tokenize(text)
            self.class_counts[label] += 1
            self.word_counts[label].update(tokens)
            self.vocabulary.update(tokens)

    def predict_proba(self, text):
        """Return {label: probability} for the given text, highest first."""
        if not self.class_counts:
            raise ValueError("Classifier has not been trained")

        tokens = tokenize(text)
        total_examples = sum(self.class_counts.values())
        vocab_size = len(self.vocabulary)
        log_scores = {}
        for label, count in self.class_counts.items():
            score = math.log(count / total_examples)
            label_words = self.word_counts[label]
            denominator = sum(label_words.values()) + self.alpha * vocab_size
            for token in tokens:
                score += math.log((label_words[token] + self.alpha) / denominator)
            log_scores[label] = score

        # Softmax over log scores, shifted by the max for numerical stability.
        top = max(log_scores.values())
        exp_scores = {label: math.exp(s - top) for label, s in log_scores.items()}
        norm = sum(exp_scores.values())
        probs = {label: s / norm for label, s in exp_scores.items()}
        return dict(sorted(probs.items(), key=lambda item: item[1], reverse=True))

    def predict(self, text):
        """Return (label, probability) for the most likely label."""
        return next(iter(self.predict_proba(text).items()))


class ChatBot:
    """Chatbot that maps a message to an intent and replies from that intent."""

    def __init__(self, intents_path=DEFAULT_INTENTS, threshold=0.2, seed=None):
        with open(intents_path, encoding="utf-8") as f:
            data = json.load(f)
        self.responses = {i["tag"]: i["responses"] for i in data["intents"]}
        self.classifier = NaiveBayesClassifier()
        self.classifier.train(
            (pattern, intent["tag"])
            for intent in data["intents"]
            for pattern in intent["patterns"]
        )
        self.threshold = threshold
        self.rng = random.Random(seed)

    def classify(self, message):
        """Return (intent, confidence), or (None, confidence) if unsure."""
        if not any(t in self.classifier.vocabulary for t in tokenize(message)):
            return None, 0.0
        label, confidence = self.classifier.predict(message)
        if confidence < self.threshold:
            return None, confidence
        return label, confidence

    def respond(self, message):
        intent, _ = self.classify(message)
        options = self.responses[intent] if intent else FALLBACK_RESPONSES
        return self.rng.choice(options)
