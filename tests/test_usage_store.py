import datetime

import httpx
import pytest

import usage_store
from tests.conftest import KV_REST_API_TOKEN, KV_REST_API_URL

# 오늘이 될 수 없는 날짜: 넘겨받은 날짜를 무시하고 오늘 날짜를 쓰는 구현을 잡는다
GIVEN_DAY = "2020-01-01"
PAID_KEY = f"{usage_store.PAID_MODEL_USAGE_KEY}:2026-11-07"
IMAGE_KEY = f"{usage_store.IMAGE_SEARCH_USAGE_KEY}:{GIVEN_DAY}"
ONE_DAY_SECONDS = 24 * 60 * 60


def utc(*args):
    return datetime.datetime(*args, tzinfo=datetime.timezone.utc)


def freeze_now(monkeypatch, now):
    monkeypatch.setattr(usage_store, "_now", lambda: now)


class FakeUpstash:
    """httpx.post를 대신해 요청을 기록하고 정해진 응답을 돌려준다."""

    def __init__(self, status_code=200, body=None, error=None):
        self.status_code = status_code
        self.body = [{"result": 1}, {"result": 1}] if body is None else body
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


def test_increase_paid_model_usage_counts_under_todays_key_and_sets_expiry(upstash, monkeypatch):
    freeze_now(monkeypatch, utc(2026, 11, 8, 3, 0))
    upstash.body = [{"result": 7}, {"result": 1}]

    assert usage_store.increase_paid_model_usage() == 7
    assert upstash.calls[0]["json"] == [
        ["INCRBY", PAID_KEY, 1],
        ["EXPIRE", PAID_KEY, usage_store.USAGE_KEY_TTL_SECONDS],
    ]


def test_increase_image_search_usage_counts_under_given_day_and_sets_expiry(upstash):
    upstash.body = [{"result": 12}, {"result": 1}]

    assert usage_store.increase_image_search_usage(3, GIVEN_DAY) == 12
    assert upstash.calls[0]["json"] == [
        ["INCRBY", IMAGE_KEY, 3],
        ["EXPIRE", IMAGE_KEY, usage_store.USAGE_KEY_TTL_SECONDS],
    ]


def test_decrease_image_search_usage_targets_given_day_and_keeps_expiry(upstash):
    upstash.body = [{"result": 9}, {"result": 1}]

    assert usage_store.decrease_image_search_usage(3, GIVEN_DAY) == 9
    assert upstash.calls[0]["json"] == [
        ["DECRBY", IMAGE_KEY, 3],
        ["EXPIRE", IMAGE_KEY, usage_store.USAGE_KEY_TTL_SECONDS],
    ]


@pytest.mark.parametrize(
    "now, expected",
    [
        (utc(2026, 11, 8, 7, 59, 59), "2026-11-07"),  # 16:59:59 KST
        (utc(2026, 11, 8, 8, 0, 0), "2026-11-08"),  # 17:00:00 KST
        (utc(2027, 1, 1, 7, 59, 59), "2026-12-31"),
        (utc(2027, 1, 1, 8, 0, 0), "2027-01-01"),
        # 미국 서머타임 기간에도 경계는 17시 KST로 고정이다
        (utc(2026, 7, 8, 7, 59, 59), "2026-07-07"),
        (utc(2026, 7, 8, 8, 0, 0), "2026-07-08"),
    ],
)
def test_usage_day_changes_at_1700_kst(monkeypatch, now, expected):
    freeze_now(monkeypatch, now)

    assert usage_store.current_usage_day() == expected


def test_paid_usage_starts_a_new_counter_when_the_day_changes(upstash, monkeypatch):
    freeze_now(monkeypatch, utc(2026, 11, 8, 7, 59, 59))
    usage_store.increase_paid_model_usage()
    freeze_now(monkeypatch, utc(2026, 11, 8, 8, 0, 0))
    usage_store.increase_paid_model_usage()

    assert [call["json"][0][1] for call in upstash.calls] == [
        f"{usage_store.PAID_MODEL_USAGE_KEY}:2026-11-07",
        f"{usage_store.PAID_MODEL_USAGE_KEY}:2026-11-08",
    ]


def test_current_usage_day_reads_the_real_clock():
    before = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=8)
    usage_day = usage_store.current_usage_day()
    after = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=8)

    assert usage_day in {before.date().isoformat(), after.date().isoformat()}


def test_key_does_not_expire_during_its_usage_day():
    assert usage_store.USAGE_KEY_TTL_SECONDS >= ONE_DAY_SECONDS


def test_counter_keys_are_namespaced_for_shared_database():
    assert usage_store.PAID_MODEL_USAGE_KEY.startswith("llm:")
    assert usage_store.IMAGE_SEARCH_USAGE_KEY.startswith("llm:")


def test_request_uses_transaction_url_bearer_token_and_finite_timeout(upstash):
    usage_store.increase_paid_model_usage()

    call = upstash.calls[0]
    assert call["url"] == f"{KV_REST_API_URL}/multi-exec"
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


@pytest.mark.parametrize(
    "body",
    [
        {"error": "EXECABORT"},
        [],
        ["unexpected"],
        [{"error": "WRONGTYPE"}, {"result": 1}],
        [{"result": 1}, {"error": "WRONGTYPE"}],
    ],
)
def test_response_without_every_result_raises_unavailable(upstash, body):
    upstash.body = body

    with pytest.raises(usage_store.UsageStoreUnavailableError):
        usage_store.increase_paid_model_usage()


def test_non_json_response_raises_unavailable(monkeypatch):
    def post(url, headers=None, json=None, timeout=None):
        return httpx.Response(200, text="<html>not json</html>", request=httpx.Request("POST", url))

    monkeypatch.setattr(usage_store.httpx, "post", post)

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
    assert str(raised.value) == "INCRBY 실패: HTTPStatusError"
