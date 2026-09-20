import os
import uuid
import yaml
from typing import List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from vaxguard.core.logger import get_logger
from vaxguard.models.immune import BehavioralProfile

logger = get_logger("BehavioralBaselineEngine")


class BehavioralBaselineEngine:
    """
    Learns normal model behavior from reference benign datasets and establishes
    a statistical, semantic vector baseline to detect drift and anomalies.
    """

    def __init__(self, max_features: int = 500, ngram_range: Tuple[int, int] = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            analyzer="word",
            lowercase=True,
            stop_words="english",
        )
        self.is_fitted = False
        self.profile: Optional[BehavioralProfile] = None

    def fit_from_texts(self, texts: List[str], profile_id: Optional[str] = None) -> BehavioralProfile:
        """
        Fits the TF-IDF vectorizer on benign texts and calculates the mathematical centroid
        and distance distribution thresholds.
        """
        if not texts or len(texts) == 0:
            raise ValueError("Cannot establish baseline from empty text dataset.")

        logger.info(f"Establishing behavioral baseline from {len(texts)} sample interactions...")
        
        tfidf_matrix = self.vectorizer.fit_transform(texts).toarray()
        self.is_fitted = True

        # Calculate mathematical centroid vector
        centroid = np.mean(tfidf_matrix, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        # Calculate cosine distances from centroid to all training samples
        # Cosine distance = 1.0 - cosine_similarity
        similarities = cosine_similarity(tfidf_matrix, [centroid]).flatten()
        distances = 1.0 - similarities

        mean_dist = float(np.mean(distances))
        std_dist = float(np.std(distances))
        # 3-sigma rule for 99.7% confidence upper bound threshold
        max_thresh = float(mean_dist + 3.0 * std_dist)

        avg_len = float(np.mean([len(t) for t in texts]))
        vocab_size = len(self.vectorizer.vocabulary_)

        self.profile = BehavioralProfile(
            profile_id=profile_id or f"profile_{uuid.uuid4().hex[:8]}",
            version=1,
            sample_count=len(texts),
            centroid_vector=centroid.tolist(),
            mean_distance=mean_dist,
            std_distance=std_dist,
            max_threshold_distance=max_thresh,
            avg_prompt_length=avg_len,
            vocabulary_size=vocab_size,
        )

        logger.info(
            f"Baseline '{self.profile.profile_id}' established: "
            f"mean_dist={mean_dist:.4f}, std_dist={std_dist:.4f}, threshold={max_thresh:.4f}, vocab={vocab_size}"
        )
        return self.profile

    def fit_from_yaml(self, yaml_path: Optional[str] = None) -> BehavioralProfile:
        """Loads benign samples from YAML file and builds the baseline."""
        if yaml_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            yaml_path = os.path.join(project_root, "data", "benign.yaml")

        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Benign dataset file not found at: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or []

        texts = [item["payload"] for item in data if isinstance(item, dict) and "payload" in item]
        if not texts:
            raise ValueError(f"No valid payloads found in {yaml_path}")

        return self.fit_from_texts(texts)

    def transform_text(self, text: str) -> np.ndarray:
        """Transforms a single text into the baseline vector space."""
        if not self.is_fitted:
            raise RuntimeError("Baseline engine has not been fitted. Call fit_from_texts or fit_from_yaml first.")
        return self.vectorizer.transform([text]).toarray()[0]
