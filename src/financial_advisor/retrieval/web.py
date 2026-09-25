"""Retrieve ranked candidates from Advisor-selected web research."""

from collections.abc import Sequence
from hashlib import sha256
from urllib.parse import urlsplit

from pypdf.errors import PdfReadError

from financial_advisor.config import (
    DEFAULT_WEB_MAX_PAGE_BYTES,
    DEFAULT_WEB_RESULT_LIMIT,
    DEFAULT_WEB_SELECTED_PASSAGE_LIMIT,
    DEFAULT_WEB_TIMEOUT_SECONDS,
    DEFAULT_WEB_USABLE_PAGE_TARGET,
    WEB_CHUNK_MAX_TOKENS,
    WEB_CHUNK_MIN_TOKENS,
    WEB_CHUNK_OVERLAP_TOKENS,
)
from financial_advisor.contracts import RetrievalChannel, RetrievedEvidenceCandidate
from financial_advisor.retrieval.documents import (
    Chunk,
    Decode,
    Encode,
    chunk_document,
    parse_document,
)
from financial_advisor.retrieval.providers import (
    SearchProvider,
    WebDiscovery,
    WebError,
    WebResult,
)
from financial_advisor.retrieval.ranking import Embed, cosine_similarity
from financial_advisor.retrieval.web_fetch import WebPageFetcher


class WebCandidateRetriever:
    """Discover, fetch, chunk, and rank web evidence candidates."""

    def __init__(
        self,
        providers: Sequence[tuple[str, SearchProvider]],
        encode: Encode,
        decode: Decode,
        embed: Embed,
        *,
        result_limit: int = DEFAULT_WEB_RESULT_LIMIT,
        usable_page_target: int = DEFAULT_WEB_USABLE_PAGE_TARGET,
        selected_passage_limit: int = DEFAULT_WEB_SELECTED_PASSAGE_LIMIT,
        timeout_seconds: float = DEFAULT_WEB_TIMEOUT_SECONDS,
        max_page_bytes: int = DEFAULT_WEB_MAX_PAGE_BYTES,
    ) -> None:
        if min(usable_page_target, selected_passage_limit) <= 0:
            raise ValueError("Web retrieval limits must be positive.")
        self.discovery = WebDiscovery(providers, result_limit=result_limit)
        self.page_fetcher = WebPageFetcher(
            timeout_seconds=timeout_seconds,
            max_page_bytes=max_page_bytes,
        )
        self.encode = encode
        self.decode = decode
        self.embed = embed
        self.usable_page_target = usable_page_target
        self.selected_passage_limit = selected_passage_limit

    def search(self, query: str) -> tuple[list[RetrievedEvidenceCandidate], list[str]]:
        """Return ranked web candidates and deterministic limitations."""

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Web query must not be blank.")

        try:
            discovered_web_results = self.discovery.search(normalized_query)
        except WebError:
            return [], ["Web research was unavailable."]

        web_passages = self._fetch_and_chunk_pages(discovered_web_results)
        if not web_passages:
            return [], ["Web research returned no usable pages."]
        return self._rank_passages(normalized_query, web_passages), []

    def _fetch_and_chunk_pages(
        self,
        discovered_web_results: Sequence[WebResult],
    ) -> list[Chunk]:
        web_passages: list[Chunk] = []
        seen_urls: set[str] = set()
        chunked_page_count = 0

        for discovered_result in discovered_web_results:
            if chunked_page_count >= self.usable_page_target:
                break
            if discovered_result.url in seen_urls:
                continue
            seen_urls.add(discovered_result.url)
            fetched_web_document = self.page_fetcher.fetch(discovered_result)
            if fetched_web_document is None:
                continue
            try:
                content_hash = sha256(fetched_web_document.content).hexdigest()
                page_chunks = chunk_document(
                    parse_document(
                        fetched_web_document.content,
                        fetched_web_document.content_type,
                    ),
                    source_id=f"web:{content_hash}",
                    source_title=fetched_web_document.title,
                    publisher=(urlsplit(fetched_web_document.url).hostname or "unknown publisher"),
                    source_url=fetched_web_document.url,
                    topics=(RetrievalChannel.WEB.value,),
                    content_hash=content_hash,
                    encode=self.encode,
                    decode=self.decode,
                    max_tokens=WEB_CHUNK_MAX_TOKENS,
                    overlap_tokens=WEB_CHUNK_OVERLAP_TOKENS,
                    min_tokens=WEB_CHUNK_MIN_TOKENS,
                )
            except (PdfReadError, ValueError):
                # Invalid source content is unusable, so continue with other pages.
                continue
            if page_chunks:
                web_passages.extend(page_chunks)
                chunked_page_count += 1
        return web_passages

    def _rank_passages(
        self,
        normalized_query: str,
        web_passages: Sequence[Chunk],
    ) -> list[RetrievedEvidenceCandidate]:
        query_and_passage_vectors = self.embed(
            [
                normalized_query,
                *(passage.embedding_text for passage in web_passages),
            ]
        )
        if len(query_and_passage_vectors) != len(web_passages) + 1:
            raise ValueError("Embedder must return one vector per web passage.")

        query_vector = query_and_passage_vectors[0]
        passage_vectors = query_and_passage_vectors[1:]
        scored_web_passages = [
            (
                passage,
                vector,
                cosine_similarity(query_vector, vector),
            )
            for passage, vector in zip(web_passages, passage_vectors, strict=True)
        ]
        top_web_passages = sorted(
            scored_web_passages,
            key=lambda item: (
                -item[2],
                item[0].canonical_candidate_id,
            ),
        )[: self.selected_passage_limit]
        return [
            RetrievedEvidenceCandidate.model_validate(
                {
                    "chunk_id": passage.chunk_id,
                    "canonical_candidate_id": passage.canonical_candidate_id,
                    "source_id": passage.source_id,
                    "source_title": passage.source_title,
                    "publisher": passage.publisher,
                    "source_url": passage.source_url,
                    "topics": passage.topics,
                    "heading_path": passage.heading_path,
                    "section_position": passage.section_position,
                    "chunk_position": passage.chunk_position,
                    "token_count": passage.token_count,
                    "text": passage.text,
                    "retrieval_channel": RetrievalChannel.WEB,
                    "rank": rank,
                    "cosine_distance": max(0.0, 1 - similarity),
                    "vector": vector,
                }
            )
            for rank, (passage, vector, similarity) in enumerate(
                top_web_passages,
                start=1,
            )
        ]
