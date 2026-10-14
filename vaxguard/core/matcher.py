"""
VaxGuard Core Cascaded Multi-Tier Matcher.

Provides ultra-fast, zero-network-dependency security matching combining:
1. Stage 1: Compiled Regex & Exact Pattern Matcher (~0.05ms)
2. Stage 2: Sparse Lexical Matcher (BM25 & Token Inverted Index, ~0.15ms)
3. Stage 3: Hashed Feature Dot-Product Vector Matcher (~0.35ms)

Target Latency: P50 < 0.2ms local evaluation on commodity CPU hardware.
No external network calls, no heavy transformer models in the hot path.
"""

from __future__ import annotations

from vaxguard.nemo.hybrid_matcher import (
    EngineType,
    GuardrailAction,
    MatchResult,
    GuardrailPattern,
    normalize_adversarial_text,
    tokenize_words,
    get_char_ngrams,
    HeuristicMatcher,
    SparseLexicalMatcher,
    FastLocalEmbedder,
    DenseSemanticVectorMatcher,
    TriHybridMatcher,
)

# Semantic aliases
LexicalBM25Matcher = SparseLexicalMatcher
DenseSemanticMatcher = DenseSemanticVectorMatcher

__all__ = [
    "EngineType",
    "GuardrailAction",
    "MatchResult",
    "GuardrailPattern",
    "normalize_adversarial_text",
    "tokenize_words",
    "get_char_ngrams",
    "HeuristicMatcher",
    "SparseLexicalMatcher",
    "FastLocalEmbedder",
    "DenseSemanticVectorMatcher",
    "TriHybridMatcher",
    "LexicalBM25Matcher",
    "DenseSemanticMatcher",
]
