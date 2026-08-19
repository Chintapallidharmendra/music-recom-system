"""API contract tests against contracts/openapi_notes.md: the four originally-frozen
routes, correct request/response shapes, no undocumented params beyond the documented
/admin/reload-policy operational addition (see contracts/openapi_notes.md's addendum).
Kafka is expected to be unreachable in this environment -- these tests exercise the
soft-fail path, not the Kafka round-trip (covered by PROJECT_PLAN.md's smoke sequence).
"""
import pytest
from fastapi.testclient import TestClient

from service.main import app

EXPECTED_ROUTES = {"/health", "/recommend", "/feedback", "/metrics"}
ADMIN_ROUTES = {"/admin/reload-policy"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs startup/shutdown lifespan events
        yield c


def test_only_four_documented_routes():
    paths = {route.path for route in app.routes if hasattr(route, "methods")}
    assert EXPECTED_ROUTES <= paths
    # nothing beyond our four routes, the documented admin addition, and FastAPI's own
    # docs/openapi/redoc endpoints
    extra = paths - EXPECTED_ROUTES - ADMIN_ROUTES
    assert extra <= {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}


def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "status" in resp.json()


def test_recommend_returns_track_id(client):
    resp = client.post("/recommend", json={"user_id": "user_000001"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"track_id"}
    assert isinstance(body["track_id"], str)


def test_recommend_rejects_missing_user_id(client):
    resp = client.post("/recommend", json={})
    assert resp.status_code == 422  # pydantic validation error, not a 500


def test_feedback_accepts_valid_action(client):
    resp = client.post(
        "/feedback",
        json={"user_id": "user_000001", "track_id": "2", "action": "liked"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_feedback_rejects_invalid_action(client):
    resp = client.post(
        "/feedback",
        json={"user_id": "user_000001", "track_id": "2", "action": "not_a_real_action"},
    )
    assert resp.status_code == 422


def test_metrics_shape(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"ctr", "avg_reward", "cumulative_regret", "total_recommendations"}


def test_cold_start_user_does_not_500(client):
    resp = client.post("/recommend", json={"user_id": "user_never_seen_before"})
    assert resp.status_code == 200
