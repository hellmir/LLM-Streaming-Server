import fastapi.testclient
import pytest

import app as app_module
import routers
import services
import usage_store
from tests.conftest import FRONTEND_URL, SECRET_KEY, SECRET_KEY_FOR_PAID_MODEL


@pytest.fixture
def client():
    return fastapi.testclient.TestClient(app_module.app)


@pytest.fixture
def llm_calls(monkeypatch):
    calls = []

    def fake_generate_response(llm_type, template, options):
        calls.append(llm_type)
        yield "안녕"
        yield "하세요"

    monkeypatch.setattr(services, "generate_response", fake_generate_response)
    return calls


@pytest.fixture
def paid_usage(monkeypatch):
    """유료 모델 사용량을 정해진 값으로 돌려주고 호출 횟수를 센다."""

    class Counter:
        value = 1
        calls = 0

    def increase():
        Counter.calls += 1
        return Counter.value

    monkeypatch.setattr(usage_store, "increase_paid_model_usage", increase)
    return Counter


def streaming_body(llm_type, secret_key):
    return {"llm_type": llm_type, "template": "인사해 줘", "options": {}, "secret_key": secret_key}


@pytest.mark.parametrize("llm_type", ["gpt", "claude", "deepseek"])
def test_paid_model_within_limit_streams(client, llm_calls, paid_usage, llm_type):
    response = client.post("/streaming/sse", json=streaming_body(llm_type, SECRET_KEY_FOR_PAID_MODEL))

    assert response.status_code == 200
    assert "data: 안녕" in response.text
    assert llm_calls == [llm_type]
    assert paid_usage.calls == 1


def test_paid_model_at_exact_limit_is_allowed(client, llm_calls, paid_usage):
    paid_usage.value = 10_000

    response = client.post("/streaming/sse", json=streaming_body("gpt", SECRET_KEY_FOR_PAID_MODEL))

    assert response.status_code == 200
    assert llm_calls == ["gpt"]


def test_paid_model_over_limit_is_rejected_without_calling_llm(client, llm_calls, paid_usage):
    paid_usage.value = 10_001

    response = client.post("/streaming/sse", json=streaming_body("gpt", SECRET_KEY_FOR_PAID_MODEL))

    assert response.status_code == 403
    assert "한도 초과" in response.json()["detail"]
    assert llm_calls == []


def test_paid_model_with_wrong_key_is_rejected_without_counting(client, llm_calls, paid_usage):
    response = client.post("/streaming/sse", json=streaming_body("gpt", SECRET_KEY))

    assert response.status_code == 401
    assert paid_usage.calls == 0
    assert llm_calls == []


def test_paid_model_is_rejected_when_usage_store_is_down(client, llm_calls, monkeypatch):
    def down():
        raise usage_store.UsageStoreUnavailableError("down")

    monkeypatch.setattr(usage_store, "increase_paid_model_usage", down)

    response = client.post("/streaming/sse", json=streaming_body("gpt", SECRET_KEY_FOR_PAID_MODEL))

    assert response.status_code == 503
    assert llm_calls == []


@pytest.mark.parametrize("llm_type", ["mistral", "default", "", None, "clovax", "gemini", "llama"])
def test_free_model_streams_without_touching_usage_store(client, llm_calls, paid_usage, llm_type):
    response = client.post("/streaming/sse", json=streaming_body(llm_type, SECRET_KEY))

    assert response.status_code == 200
    assert paid_usage.calls == 0
    assert len(llm_calls) == 1


def test_free_model_with_wrong_key_is_rejected(client, llm_calls):
    response = client.post("/streaming/sse", json=streaming_body("mistral", "wrong"))

    assert response.status_code == 401
    assert llm_calls == []


def test_plain_streaming_endpoint_returns_chunks(client, llm_calls):
    response = client.post("/streaming", json=streaming_body("mistral", SECRET_KEY))

    assert response.status_code == 200
    assert response.text == "안녕하세요"


def test_get_sse_endpoint_streams_with_query_parameters(client, llm_calls):
    response = client.get("/streaming/sse", params={"llm_type": "mistral", "template": "인사", "secret_key": SECRET_KEY})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data: 안녕" in response.text


def test_sync_endpoint_returns_llm_result(client, monkeypatch):
    async def fake_generate_sync_response(llm_type, template, options):
        return [{"name": "토마토 파스타"}]

    monkeypatch.setattr(services, "generate_sync_response", fake_generate_sync_response)

    response = client.post("/sync", json=streaming_body("mistral", SECRET_KEY))

    assert response.status_code == 200
    assert response.json() == [{"name": "토마토 파스타"}]


def test_docs_and_openapi_are_served(client):
    assert client.get("/docs").status_code == 200

    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    assert "/streaming/sse" in openapi.json()["paths"]


def test_daily_reset_endpoint_is_removed(client):
    assert client.get("/api/cron/reset-daily-usage").status_code == 404


def test_preflight_allows_configured_frontend(client):
    response = client.options(
        "/streaming/sse",
        headers={
            "Origin": FRONTEND_URL,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == FRONTEND_URL


def test_preflight_rejects_other_origin(client):
    response = client.options(
        "/streaming/sse",
        headers={"Origin": "https://evil.test", "Access-Control-Request-Method": "POST"},
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_in_memory_usage_globals_are_gone():
    assert not hasattr(routers, "paid_model_usage")
    assert not hasattr(routers, "reset_gpt_usage")
    assert not hasattr(services, "image_search_usage")
    assert not hasattr(services, "reset_image_search_usage")
