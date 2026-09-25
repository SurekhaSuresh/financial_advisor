"""Small deterministic vector operations shared by retrieval and diversity selection."""

from collections.abc import Sequence
from math import sqrt


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity and reject malformed or zero vectors clearly."""

    if len(left) != len(right):
        raise ValueError("Candidate vectors must have the same dimension.")
    left_magnitude_squared = sum(value * value for value in left)
    right_magnitude_squared = sum(value * value for value in right)
    if left_magnitude_squared == 0 or right_magnitude_squared == 0:
        raise ValueError("Candidate vectors must not be zero vectors.")
    dot_product = sum(
        left_value * right_value for left_value, right_value in zip(left, right, strict=True)
    )
    return dot_product / sqrt(left_magnitude_squared * right_magnitude_squared)
