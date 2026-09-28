"""Local tokenization, embedding, and reranking for retrieval."""

from collections.abc import Sequence
from pathlib import Path

from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from transformers import AutoTokenizer

from financial_advisor.config import EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME


class LocalRetrievalModels:
    """Load and share the local models required by retrieval."""

    def __init__(self, cache_directory: Path) -> None:
        cache_directory.mkdir(parents=True, exist_ok=True)
        self._embedding_model = TextEmbedding(
            model_name=EMBEDDING_MODEL_NAME,
            cache_dir=str(cache_directory),
        )
        self._reranker_model = TextCrossEncoder(
            model_name=RERANKER_MODEL_NAME,
            cache_dir=str(cache_directory),
        )
        self._tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            EMBEDDING_MODEL_NAME,
            cache_dir=str(cache_directory),
        )

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: Sequence[int]) -> str:
        return str(self._tokenizer.decode(token_ids, skip_special_tokens=True)).strip()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._embedding_model.embed(texts)]

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        return [float(score) for score in self._reranker_model.rerank(query, documents)]
