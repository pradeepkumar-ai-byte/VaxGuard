import pytest
from vaxguard.immune.baseline import BehavioralBaselineEngine
from vaxguard.models.immune import BehavioralProfile


def test_baseline_fit_from_texts():
    texts = [
        "What is the capital of France?",
        "How do I sort a list in Python?",
        "Can you explain the theory of relativity?",
        "Write a polite email requesting a meeting.",
        "Translate this sentence into Spanish.",
    ]
    engine = BehavioralBaselineEngine()
    profile = engine.fit_from_texts(texts)

    assert isinstance(profile, BehavioralProfile)
    assert profile.sample_count == len(texts)
    assert len(profile.centroid_vector) > 0
    assert profile.mean_distance >= 0.0
    assert profile.max_threshold_distance > profile.mean_distance
    assert engine.is_fitted is True


def test_baseline_transform():
    texts = ["Simple query one", "Simple query two"]
    engine = BehavioralBaselineEngine()
    engine.fit_from_texts(texts)

    vec = engine.transform_text("Simple query test")
    assert len(vec) == len(engine.profile.centroid_vector)


def test_baseline_empty_raises():
    engine = BehavioralBaselineEngine()
    with pytest.raises(ValueError):
        engine.fit_from_texts([])
