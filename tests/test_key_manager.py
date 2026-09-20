from vaxguard.core.key_manager import GroqKeyManager

def test_round_robin():
    keys = ["key1", "key2", "key3"]
    manager = GroqKeyManager(keys=keys)
    
    assert manager.get_next_key() == "key1"
    assert manager.get_next_key() == "key2"
    assert manager.get_next_key() == "key3"
    assert manager.get_next_key() == "key1" # Cycles back correctly
