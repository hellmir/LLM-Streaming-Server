import httpx
import pytest

import usage_store
from tests.conftest import KV_REST_API_TOKEN, KV_REST_API_URL


class FakeUpstash:
    """httpx.post를 대신해 요청을 기록하고 정해진 응답을 돌려준다."""

    def __init__(self, status_code=200, body=None, error=None):
        self.status_code = status_code
        self.body = {"result": 1} if body is None else body
        self.error = error
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        if self.error:
            raise self.error
        return httpx.Response(self.status_code, json=self.body, request=httpx.Request("POST", url))


@pytest.fixture
def upstash(monkeypatch):
    fake = FakeUpstash()
    monkeypatch.setattr(usage_store.httpx, "post", fake.post)
    return fake


def test_increase_paid_model_usage_sends_incr_and_returns_count(upstash):
    upstash.body = {"result": 7}

    assert usage_store.increase_paid_model_usage() == 7
    assert upstash.calls[0]["json"] == ["INCR", usage_store.PAID_MODEL_USAGE_KEY]


def test_increase_image_search_usage_sends_incrby_with_count(upstash):
    upstash.body = {"result": 12}

    assert usage_store.increase_image_search_usage(3) == 12
    assert upstash.calls[0]["json"] == ["INCRBY", usage_store.IMAGE_SEARCH_USAGE_KEY, 3]


def test_decrease_image_search_usage_sends_decrby_with_count(upstash):
    usage_store.decrease_image_search_usage(3)

    assert upstash.calls[0]["json"] == ["DECRBY", usage_store.IMAGE_SEARCH_USAGE_KEY, 3]


def test_reset_daily_usage_deletes_both_counters_in_one_command(upstash):
    usage_store.reset_daily_usage()

    assert len(upstash.calls) == 1
    assert upstash.calls[0]["json"] == ["DEL", usage_store.PAID_MODEL_USAGE_KEY, usage_store.IMAGE_SEARCH_USAGE_KEY]


def test_counter_keys_are_namespaced_for_shared_database():
    assert usage_store.PAID_MODEL_USAGE_KEY.startswith("llm:")
    assert usage_store.IMAGE_SEARCH_USAGE_KEY.startswith("llm:")


def test_request_uses_rest_url_bearer_token_and_finite_timeout(upstash):
    usage_store.increase_paid_model_usage()

    call = upstash.calls[0]
    assert call["url"] == KV_REST_API_URL
    assert call["headers"] == {"Authorization": f"Bearer {KV_REST_API_TOKEN}"}
    assert 0 < call["timeout"] <= 5


@pytest.mark.parametrize("missing", ["KV_REST_API_URL", "KV_REST_API_TOKEN"])
def test_missing_connection_env_raises_unavailable_without_http_call(upstash, monkeypatch, missing):
    monkeypatch.delenv(missing)

    with pytest.raises(usage_store.UsageStoreUnavailableError):
        usage_store.increase_paid_model_usage()
    assert upstash.calls == []


@pytest.mark.parametrize("status_code", [400, 401, 500, 503])
def test_error_status_raises_unavailable(upstash, status_code):
    upstash.status_code = status_code
    upstash.body = {"error": "failed"}

    with pytest.raises(usage_store.UsageStoreUnavailableError):
        usage_store.increase_paid_model_usage()


@pytest.mark.parametrize("error", [httpx.ConnectTimeout("timeout"), httpx.ConnectError("refused")])
def test_network_failure_raises_unavailable(upstash, error):
    upstash.error = error

    with pytest.raises(usage_store.UsageStoreUnavailableError):
        usage_store.increase_paid_model_usage()


def test_response_without_result_raises_unavailable(upstash):
    upstash.body = {"error": "WRONGTYPE"}

    with pytest.raises(usage_store.UsageStoreUnavailableError):
        usage_store.increase_paid_model_usage()


def test_non_json_response_raises_unavailable(monkeypatch):
    def post(url, headers=None, json=None, timeout=None):
        return httpx.Response(200, text="<html>not json</html>", request=httpx.Request("POST", url))

    monkeypatch.setattr(usage_store.httpx, "post", post)

    with pytest.raises(usage_store.UsageStoreUnavailableError):
        usage_store.increase_paid_model_usage()


def test_non_object_json_response_raises_unavailable(upstash):
    upstash.body = ["unexpected"]

    with pytest.raises(usage_store.UsageStoreUnavailableError):
        usage_store.increase_paid_model_usage()


def test_malformed_rest_url_raises_unavailable(monkeypatch):
    # 실제 httpx.post를 호출한다: URL 해석 단계에서 실패하므로 네트워크로 나가지 않는다
    monkeypatch.setenv("KV_REST_API_URL", "https://kv.test:not-a-port")

    with pytest.raises(usage_store.UsageStoreUnavailableError):
        usage_store.increase_paid_model_usage()


def test_error_message_only_names_the_command_and_failure_type(upstash):
    # 응답 본문이나 요청 정보가 로그로 새지 않도록 메시지를 고정한다
    upstash.status_code = 401
    upstash.body = {"error": f"unauthorized {KV_REST_API_TOKEN}"}

    with pytest.raises(usage_store.UsageStoreUnavailableError) as raised:
        usage_store.increase_paid_model_usage()
    assert str(raised.value) == "INCR 실패: HTTPStatusError"
