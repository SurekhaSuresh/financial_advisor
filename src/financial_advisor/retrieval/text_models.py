"""Local tokenization, embedding, and reranking for retrieval."""

from collections.abc import Sequence
from pathlib import Path
from typing import cast

from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from financial_advisor.config import (
    EMBEDDING_MODEL_NAME,
    RERANKER_MODEL_NAME,
    TOKENIZER_MODEL_MAX_LENGTH,
)


class LocalRetrievalModels:
    """Load and share the local models required by retrieval."""

    def __init__(
        self,
        cache_directory: Path,
        *,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        reranker_model_name: str = RERANKER_MODEL_NAME,
    ) -> None:
        cache_directory.mkdir(parents=True, exist_ok=True)
        self._embedding_model = TextEmbedding(
            model_name=embedding_model_name,
            cache_dir=str(cache_directory),
        )
        self._reranker_model = TextCrossEncoder(
            model_name=reranker_model_name,
            cache_dir=str(cache_directory),
        )
        tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            embedding_model_name,
            cache_dir=str(cache_directory),
        )
        self._tokenizer = cast(PreTrainedTokenizerBase, tokenizer)
        self._tokenizer.model_max_length = TOKENIZER_MODEL_MAX_LENGTH

    def encode(self, text: str) -> list[int]:
        return cast(list[int], self._tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._embedding_model.embed(texts)]

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        return [float(score) for score in self._reranker_model.rerank(query, documents)]
