"""Build ranked evidence candidates from discovered web sources."""

from hashlib import sha256
from urllib.parse import urlsplit

from pypdf.errors import PdfReadError

from financial_advisor.config import (
    DEFAULT_WEB_SELECTED_PASSAGE_LIMIT,
    DEFAULT_WEB_USABLE_PAGE_TARGET,
)
from financial_advisor.contracts import DocumentChunk, RetrievalChannel, RetrievedEvidenceCandidate
from financial_advisor.retrieval.ranking import cosine_similarity
from financial_advisor.retrieval.text_models import LocalRetrievalModels
from financial_advisor.retrieval.web.content_processing import chunk_document
from financial_advisor.retrieval.web.discovery import WebSourceDiscovery
from financial_advisor.retrieval.web.page_fetch import fetch_page


class WebCandidateRetriever:
    """Discover, fetch, chunk, and rank web evidence candidates."""

    def __init__(self, models: LocalRetrievalModels, source_discovery: WebSourceDiscovery) -> None:
        self.models, self.source_discovery = models, source_discovery

    def search(self, query: str) -> list[RetrievedEvidenceCandidate]:
        """Return ranked candidates from usable web pages."""

        query = query.strip()
        if not query:
            raise ValueError("Web query must not be blank.")

        passages: list[DocumentChunk] = []
        usable_pages = 0

        for title, url in self.source_discovery.discover_source_urls(query):
            if usable_pages >= DEFAULT_WEB_USABLE_PAGE_TARGET:
                break
            if (page := fetch_page(url)) is None:
                continue
            final_url, content_type, content = page
            try:
                chunks = chunk_document(
                    content,
                    content_type,
                    source_title=title,
                    publisher=urlsplit(final_url).hostname or "unknown publisher",
                    source_url=final_url,
                    topics=(RetrievalChannel.WEB.value,),
                    content_hash=sha256(content).hexdigest(),
                    models=self.models,
                )
            except (PdfReadError, ValueError):
                # Invalid source content is unusable, so continue with other pages.
                continue
            if chunks:
                passages.extend(chunks)
                usable_pages += 1

        if not passages:
            return []

        query_vector, *passage_vectors = self.models.embed(
            [query, *(passage.embedding_text for passage in passages)]
        )
        scored_passages = [
            (passage, vector, 1 - cosine_similarity(query_vector, vector))
            for passage, vector in zip(passages, passage_vectors, strict=True)
        ]
        scored_passages.sort(key=lambda item: (item[2], item[0].canonical_candidate_id))
        return [
            RetrievedEvidenceCandidate(
                **passage.model_dump(),
                retrieval_channel=RetrievalChannel.WEB,
                rank=rank,
                cosine_distance=distance,
                vector=vector,
            )
            for rank, (passage, vector, distance) in enumerate(
                scored_passages[:DEFAULT_WEB_SELECTED_PASSAGE_LIMIT], start=1
            )
        ]
