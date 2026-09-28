from pathlib import Path

import lancedb

from financial_advisor.inspection import inspect_knowledge_chunks


def test_knowledge_inspection_filters_metadata_without_exposing_vectors(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "knowledge"
    database = lancedb.connect(str(database_path))
    database.create_table(
        "knowledge_chunks",
        data=[
            {
                "canonical_candidate_id": "candidate-1",
                "source_title": "Investment basics",
                "publisher": "FINRA",
                "source_url": "https://www.finra.org/",
                "topics": ["investing", "risk"],
                "section_position": 1,
                "chunk_position": 1,
                "token_count": 42,
                "text": "Diversification can help manage concentration risk.",
                "vector": [0.1, 0.2],
            },
            {
                "canonical_candidate_id": "candidate-2",
                "source_title": "Savings guidance",
                "publisher": "Investor.gov",
                "source_url": "https://www.investor.gov/",
                "topics": ["savings"],
                "section_position": 1,
                "chunk_position": 1,
                "token_count": 30,
                "text": "Match savings choices to the time horizon.",
                "vector": [0.3, 0.4],
            },
        ],
    )
    inspection = inspect_knowledge_chunks(
        database_path,
        publisher="finra",
        topic="risk",
        metadata_field="source_title",
        metadata_value="investment",
        min_token_count=40,
    )

    assert inspection.publishers == {"FINRA", "Investor.gov"}
    assert inspection.total_matching_chunks == 1
    assert inspection.chunks[0].canonical_candidate_id == "candidate-1"
    assert "vector" not in inspection.chunks[0].model_dump()
