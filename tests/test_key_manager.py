import threading
import pytest
from vaxguard.core.key_manager import GroqKeyManager


def test_round_robin():
    keys = ["key1", "key2", "key3"]
    manager = GroqKeyManager(keys=keys)

    assert manager.get_next_key() == "key1"
    assert manager.get_next_key() == "key2"
    assert manager.get_next_key() == "key3"
    assert manager.get_next_key() == "key1"  # Cycles back


def test_single_key():
    manager = GroqKeyManager(keys=["single_key"])
    for _ in range(5):
        assert manager.get_next_key() == "single_key"


def test_empty_keys_fallback():
    manager = GroqKeyManager(keys=[])
    assert manager.get_next_key() == "dummy_key_for_tests"


def test_cycles_multiple_loops():
    keys = ["a", "b"]
    manager = GroqKeyManager(keys=keys)
    result = [manager.get_next_key() for _ in range(6)]
    assert result == ["a", "b", "a", "b", "a", "b"]


def test_key_manager_thread_safety():
    keys = [f"key_{i}" for i in range(5)]
    manager = GroqKeyManager(keys=keys)
    retrieved = []

    def worker():
        for _ in range(50):
            k = manager.get_next_key()
            retrieved.append(k)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(retrieved) == 200
    for k in keys:
        assert k in retrieved
