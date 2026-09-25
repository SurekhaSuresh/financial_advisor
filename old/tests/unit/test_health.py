"""Smoke tests for the application foundation."""

import asyncio

import httpx

from financial_advisor.main import create_app


def test_health_endpoint_reports_application_status() -> None:
    """The API starts without agent, database, or provider dependencies."""

    async def request_health() -> httpx.Response:
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/health")

    response = asyncio.run(request_health())

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "environment": "development"}
