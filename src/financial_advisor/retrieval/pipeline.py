"""Execute selected retrieval paths and produce one trusted evidence set."""

from collections.abc import Sequence
from uuid import NAMESPACE_URL, uuid5

from financial_advisor.config import (
    DEFAULT_EVIDENCE_LIMIT,
    LOCAL_HYBRID_RETRIEVAL_PATH,
    MAX_EVIDENCE_TEXT_LENGTH,
    WEB_RETRIEVAL_PATH,
    Rerank,
)
from financial_advisor.contracts import (
    Evidence,
    RetrievedEvidenceCandidate,
)
from financial_advisor.retrieval.hybrid_search import LocalHybridRetriever
from financial_advisor.retrieval.ranking import select_evidence_candidates
from financial_advisor.retrieval.web.web_search import WebCandidateRetriever


class RetrievalPipeline:
    """Collect every enabled channel before one shared evidence-selection pass."""

    def __init__(
        self,
        local_hybrid_retriever: LocalHybridRetriever | None,
        rerank: Rerank,
        web_candidate_retriever: WebCandidateRetriever | None = None,
    ) -> None:
        self.local_hybrid_retriever = local_hybrid_retriever
        self.web_candidate_retriever = web_candidate_retriever
        self.rerank = rerank

    def retrieve(
        self,
        query: str,
        paths: Sequence[str],
    ) -> list[Evidence]:
        """Execute the Advisor's paths, then jointly select final evidence."""

        if not paths:
            raise ValueError("At least one retrieval path is required.")

        candidates: list[RetrievedEvidenceCandidate] = []
        if LOCAL_HYBRID_RETRIEVAL_PATH in paths and self.local_hybrid_retriever is not None:
            candidates.extend(self.local_hybrid_retriever.search(query))

        if WEB_RETRIEVAL_PATH in paths and self.web_candidate_retriever is not None:
            candidates.extend(self.web_candidate_retriever.search(query))

        selected_candidates = select_evidence_candidates(
            query,
            candidates,
            self.rerank,
            limit=DEFAULT_EVIDENCE_LIMIT,
        )
        return [
            Evidence(
                evidence_id=uuid5(
                    NAMESPACE_URL,
                    f"financial-advisor:{candidate.canonical_candidate_id}:{candidate.source_url}",
                ),
                title=candidate.source_title,
                publisher=candidate.publisher,
                url=candidate.source_url,
                text=candidate.text[:MAX_EVIDENCE_TEXT_LENGTH],
                source=candidate.retrieval_channel,
            )
            for candidate in selected_candidates
        ]
