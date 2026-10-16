"""SimpleAI: a tiny, dependency-free chatbot that learns intents from examples."""

from .bot import ChatBot, NaiveBayesClassifier, tokenize

__all__ = ["ChatBot", "NaiveBayesClassifier", "tokenize"]
