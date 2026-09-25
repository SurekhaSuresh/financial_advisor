"""Execute selected retrieval paths and produce one trusted evidence set."""

from collections.abc import Sequence
from uuid import NAMESPACE_URL, uuid5

from financial_advisor.config import DEFAULT_EVIDENCE_LIMIT, MAX_EVIDENCE_TEXT_LENGTH
from financial_advisor.contracts import (
    Evidence,
    Rerank,
    RetrievalPath,
    RetrievalResult,
    RetrievedEvidenceCandidate,
)
from financial_advisor.retrieval.knowledge_base.hybrid_search import LocalHybridRetriever
from financial_advisor.retrieval.ranking import select_evidence_candidates
from financial_advisor.retrieval.web.web_search import WebCandidateRetriever


class RetrievalPipeline:
    """Collect every enabled channel before one shared evidence-selection pass."""

    def __init__(
        self,
        local_hybrid_retriever: LocalHybridRetriever,
        rerank: Rerank,
        web_candidate_retriever: WebCandidateRetriever | None = None,
        *,
        evidence_limit: int = DEFAULT_EVIDENCE_LIMIT,
    ) -> None:
        if evidence_limit <= 0:
            raise ValueError("Evidence limit must be positive.")
        self.local_hybrid_retriever = local_hybrid_retriever
        self.web_candidate_retriever = web_candidate_retriever
        self.rerank = rerank
        self.evidence_limit = evidence_limit

    def retrieve(
        self,
        query: str,
        paths: Sequence[RetrievalPath],
    ) -> RetrievalResult:
        """Execute the Advisor's paths, then jointly select final evidence."""

        if not paths:
            raise ValueError("At least one retrieval path is required.")

        candidates: list[RetrievedEvidenceCandidate] = []
        limitations: list[str] = []
        if RetrievalPath.LOCAL_HYBRID in paths:
            local_candidates = self.local_hybrid_retriever.search(query)
            candidates.extend(local_candidates)
            if not local_candidates:
                limitations.append("The local knowledge store returned no usable evidence.")

        if RetrievalPath.WEB in paths:
            if self.web_candidate_retriever is None:
                limitations.append("Web research is not configured.")
            else:
                web_candidates, web_limitations = self.web_candidate_retriever.search(query)
                candidates.extend(web_candidates)
                limitations.extend(web_limitations)

        selected_candidates = select_evidence_candidates(
            query,
            candidates,
            self.rerank,
            limit=self.evidence_limit,
        )
        if not selected_candidates:
            limitations.append("Retrieval returned no evidence safe enough to use.")
        return RetrievalResult(
            evidence=[_candidate_to_evidence(candidate) for candidate in selected_candidates],
            limitations=list(dict.fromkeys(limitations)),
        )


def _candidate_to_evidence(candidate: RetrievedEvidenceCandidate) -> Evidence:
    """Create trusted evidence with a stable citation ID and bounded text."""

    return Evidence(
        evidence_id=uuid5(
            NAMESPACE_URL,
            (f"financial-advisor:{candidate.canonical_candidate_id}:{candidate.source_url}"),
        ),
        title=candidate.source_title,
        publisher=candidate.publisher,
        url=candidate.source_url,
        text=candidate.text[:MAX_EVIDENCE_TEXT_LENGTH],
        source=candidate.retrieval_channel,
    )
