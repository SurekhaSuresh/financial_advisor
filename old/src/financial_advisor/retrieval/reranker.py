"""Production cross-encoder implementation for cross-channel reranking."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fastembed.rerank.cross_encoder import TextCrossEncoder

from financial_advisor.config import get_settings

CROSS_ENCODER_MODEL_NAME = get_settings().model_runtime.cross_encoder_model_name


class FastEmbedCrossEncoderReranker:
    """Local ONNX cross-encoder implementation used by production retrieval."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._model: Any = TextCrossEncoder(
            model_name=CROSS_ENCODER_MODEL_NAME,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Return one relevance score for each supplied query-document pair."""

        return [float(score) for score in self._model.rerank(query, documents)]
