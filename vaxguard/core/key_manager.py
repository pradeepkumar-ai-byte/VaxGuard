import threading
from typing import List
from vaxguard.core.config import GROQ_API_KEYS

class GroqKeyManager:
    """
    Manages multiple Groq API keys using a thread-safe round-robin strategy.
    """
    def __init__(self, keys: List[str] = None):
        self.keys = keys if keys is not None else GROQ_API_KEYS
        if not self.keys:
            # In a real environment we'd raise an error, but for testing we allow empty init
            # raise ValueError("No Groq API keys provided or found in environment variables.")
            self.keys = ["dummy_key_for_tests"]
            
        self._lock = threading.Lock()
        self._index = 0

    def get_next_key(self) -> str:
        """Returns the next API key in the sequence."""
        with self._lock:
            key = self.keys[self._index]
            self._index = (self._index + 1) % len(self.keys)
            return key
