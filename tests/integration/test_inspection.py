"""Tests for read-only SQLite and LanceDB local inspection helpers."""

import asyncio
from decimal import Decimal
from pathlib import Path

import httpx
import lancedb
import pytest

from financial_advisor.domain import ClientMessage, ClientProfile, RiskTolerance
from financial_advisor.inspection import KnowledgeStoreInspector
from financial_advisor.main import create_app
from financial_advisor.persistence import SessionRepository
from financial_advisor.workflow import WorkflowEngine


def saved_session(engine: WorkflowEngine) -> None:
    """Persist one minimal session so the SQLite inspector has visible row values."""

    profile = ClientProfile(
        client_id="maya-chen",
        name="Maya Chen",
        age=38,
        risk_tolerance=RiskTolerance.MODERATE,
        emergency_fund_months=6,
        retirement_savings=Decimal("120000.00"),
        brokerage_savings=Decimal("35000.00"),
        student_loan_balance=Decimal("18000.00"),
        student_loan_rate_percent=Decimal("5.8"),
        primary_goal="Buy a home",
        goal_time_horizon_years=5,
    )
    session = engine.start_session(profile)
    engine.submit_opening_message(
        session,
        ClientMessage(
            session_id=session.session_id,
            text="How should I use my bonus?",
            follow_up_number=0,
        ),
    )


def test_sqlite_inspection_endpoints_expose_only_allowlisted_table_values(tmp_path: Path) -> None:
    """A UI can view stored session rows, while arbitrary SQLite tables stay unavailable."""

    repository = SessionRepository(tmp_path / "financial_advisor.db")
    saved_session(WorkflowEngine(repository))
    app = create_app(repository=repository)

    async def request_data() -> tuple[httpx.Response, httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return (
                await client.get("/inspection/sqlite/tables"),
                await client.get("/inspection/sqlite/tables/sessions/rows"),
                await client.get("/inspection/sqlite/tables/sqlite_master/rows"),
            )

    tables, rows, invalid_table = asyncio.run(request_data())

    assert tables.status_code == 200
    assert {item["table_name"] for item in tables.json()} >= {"sessions", "session_events"}
    assert rows.status_code == 200
    assert rows.json()["table_name"] == "sessions"
    assert rows.json()["rows"]
    assert invalid_table.status_code == 404


def test_lancedb_inspector_filters_human_readable_chunks(tmp_path: Path) -> None:
    """Knowledge inspection omits vectors while retaining source and topic metadata."""

    database = lancedb.connect(str(tmp_path / "lancedb"))
    database.create_table(
        "knowledge_chunks",
        data=[
            {
                "chunk_id": "finra:1:1",
                "source_id": "finra_basics",
                "source_title": "Investment basics",
                "publisher": "FINRA",
                "source_url": "https://www.finra.org/",
                "topics": ["investing", "risk"],
                "heading": "Diversification",
                "heading_path": ["Investing", "Diversification"],
                "section_position": 1,
                "chunk_position": 1,
                "token_count": 42,
                "text": "Diversification can help manage concentration risk.",
                "vector": [0.1, 0.2],
            },
            {
                "chunk_id": "irs:1:1",
                "source_id": "irs_ira",
                "source_title": "IRA guidance",
                "publisher": "IRS",
                "source_url": "https://www.irs.gov/",
                "topics": ["retirement"],
                "heading": "Contributions",
                "heading_path": ["IRA", "Contributions"],
                "section_position": 1,
                "chunk_position": 1,
                "token_count": 37,
                "text": "Contribution rules can change and should be verified.",
                "vector": [0.3, 0.4],
            },
        ],
    )
    inspector = KnowledgeStoreInspector(tmp_path / "lancedb")

    summary = inspector.summary()
    filtered = inspector.chunks(publisher="finra", topic="risk")
    exact_chunk = inspector.chunks(chunk_id="finra:1:1")
    exact_source = inspector.chunks(source_id="irs_ira")
    metadata_filtered = inspector.chunks(
        metadata_field="heading_path", metadata_value="diversification"
    )
    token_filtered = inspector.chunks(min_token_count=40)
    token_window = inspector.chunks(min_token_count=37, max_token_count=40)
    paged = inspector.chunks(limit=1, offset=1)

    assert summary.chunk_count == 2
    assert summary.publishers == {"FINRA": 1, "IRS": 1}
    assert filtered.total_matching_chunks == 1
    assert filtered.chunks[0].chunk_id == "finra:1:1"
    assert "vector" not in filtered.chunks[0].model_dump()
    assert [chunk.chunk_id for chunk in exact_chunk.chunks] == ["finra:1:1"]
    assert [chunk.chunk_id for chunk in exact_source.chunks] == ["irs:1:1"]
    assert [chunk.chunk_id for chunk in metadata_filtered.chunks] == ["finra:1:1"]
    assert [chunk.chunk_id for chunk in token_filtered.chunks] == ["finra:1:1"]
    assert [chunk.chunk_id for chunk in token_window.chunks] == ["irs:1:1"]
    assert paged.total_matching_chunks == 2
    assert paged.offset == 1
    assert [chunk.chunk_id for chunk in paged.chunks] == ["irs:1:1"]
    with pytest.raises(ValueError, match="not available"):
        inspector.chunks(metadata_field="unknown", metadata_value="value")
