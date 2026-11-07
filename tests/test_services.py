import asyncio

import httpx
import langchain_core.messages
import pytest

import services
import usage_store


KAKAO_KEY = "kakao-key-must-not-leak"


class FakeUsage:
    """이미지 검색 사용량 저장소 대역."""

    def __init__(self, after=3, increase_error=None, decrease_error=None):
        self.after = after
        self.increase_error = increase_error
        self.decrease_error = decrease_error
        self.increased = []
        self.decreased = []
        self.increased_days = []
        self.decreased_days = []

    def increase(self, count, usage_day):
        self.increased.append(count)
        self.increased_days.append(usage_day)
        if self.increase_error:
            raise self.increase_error
        return self.after

    def decrease(self, count, usage_day):
        self.decreased.append(count)
        self.decreased_days.append(usage_day)
        if self.decrease_error:
            raise self.decrease_error
        return self.after - count


@pytest.fixture
def searches(monkeypatch):
    calls = []

    async def fake_search(food_name):
        calls.append(food_name)
        return [f"https://image.test/{food_name}.png"]

    monkeypatch.setattr(services, "search_images", fake_search)
    return calls


@pytest.fixture
def default_image(monkeypatch):
    monkeypatch.setattr(services, "DEFAULT_IMAGE_URL", "https://image.test/default.png")
    return "https://image.test/default.png"


def use_search_api(monkeypatch, handler, needed_count="2"):
    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(services, "KAKAO_REST_API_KEY", KAKAO_KEY)
    monkeypatch.setenv("NEEDED_IMAGE_COUNT", needed_count)
    monkeypatch.setattr(services.httpx, "AsyncClient", lambda: real_client(transport=transport))


def documents(*image_urls, thumbnail="https://search1.kakaocdn.net/thumb.png"):
    return {"documents": [{"image_url": url, "thumbnail_url": thumbnail} for url in image_urls]}


def use_store(monkeypatch, fake):
    monkeypatch.setattr(usage_store, "increase_image_search_usage", fake.increase)
    monkeypatch.setattr(usage_store, "decrease_image_search_usage", fake.decrease)


def test_add_images_within_limit_searches_and_keeps_reservation(monkeypatch, searches):
    fake = FakeUsage(after=services.IMAGE_SEARCH_DAILY_LIMIT - 1)
    use_store(monkeypatch, fake)
    menu = [{"name": "파스타"}, {"name": "샐러드"}]

    asyncio.run(services.add_images(["파스타", "샐러드"], menu))

    assert searches == ["파스타", "샐러드"]
    assert fake.increased == [2]
    assert fake.decreased == []
    assert menu[0]["imageUrls"] == ["https://image.test/파스타.png"]
    assert menu[1]["imageUrls"] == ["https://image.test/샐러드.png"]


def test_add_images_at_limit_uses_default_image_and_rolls_back(monkeypatch, searches, default_image):
    fake = FakeUsage(after=services.IMAGE_SEARCH_DAILY_LIMIT)
    use_store(monkeypatch, fake)
    menu = [{"name": "파스타"}, {"name": "샐러드"}]

    asyncio.run(services.add_images(["파스타", "샐러드"], menu))

    assert searches == []
    assert fake.increased == [2]
    assert fake.decreased == [2]
    assert menu[0]["imageUrls"] == [default_image]
    assert menu[1]["imageUrls"] == [default_image]


def test_rollback_targets_the_reserved_day_even_if_the_day_changes_meanwhile(monkeypatch, searches, default_image):
    fake = FakeUsage(after=services.IMAGE_SEARCH_DAILY_LIMIT)
    use_store(monkeypatch, fake)
    days = iter(["2026-11-07", "2026-11-08"])
    monkeypatch.setattr(usage_store, "current_usage_day", lambda: next(days))

    asyncio.run(services.add_images(["파스타"], [{"name": "파스타"}]))

    assert fake.increased_days == ["2026-11-07"]
    assert fake.decreased_days == ["2026-11-07"]


def test_reservation_and_rollback_reach_the_real_store_with_the_same_key(monkeypatch, searches, default_image):
    requests = []
    limit = services.IMAGE_SEARCH_DAILY_LIMIT
    responses = iter([[{"result": limit}, {"result": 1}], [{"result": limit - 1}, {"result": 1}]])

    def post(url, headers=None, json=None, timeout=None):
        requests.append(json)
        return httpx.Response(200, json=next(responses), request=httpx.Request("POST", url))

    monkeypatch.setattr(usage_store.httpx, "post", post)

    urls = asyncio.run(services.get_image_urls(["파스타"]))

    assert urls == [[default_image]]
    assert [request[0][0] for request in requests] == ["INCRBY", "DECRBY"]
    assert requests[0][0][1:] == requests[1][0][1:]
    assert requests[0][0][2] == 1


def test_add_images_uses_default_image_when_store_is_down(monkeypatch, searches, default_image):
    fake = FakeUsage(increase_error=usage_store.UsageStoreUnavailableError("down"))
    use_store(monkeypatch, fake)
    menu = [{"name": "파스타"}]

    asyncio.run(services.add_images(["파스타"], menu))

    assert searches == []
    assert fake.decreased == []
    assert menu[0]["imageUrls"] == [default_image]


def test_add_images_uses_default_image_when_rollback_fails(monkeypatch, searches, default_image):
    fake = FakeUsage(after=services.IMAGE_SEARCH_DAILY_LIMIT, decrease_error=usage_store.UsageStoreUnavailableError("down"))
    use_store(monkeypatch, fake)
    menu = [{"name": "파스타"}]

    asyncio.run(services.add_images(["파스타"], menu))

    assert searches == []
    assert menu[0]["imageUrls"] == [default_image]


def test_limit_is_logged_as_quota_exhausted(monkeypatch, searches, default_image, caplog):
    use_store(monkeypatch, FakeUsage(after=services.IMAGE_SEARCH_DAILY_LIMIT))

    with caplog.at_level("WARNING"):
        asyncio.run(services.add_images(["파스타"], [{"name": "파스타"}]))

    assert services.SEARCH_USAGE_LIMIT_PER_DAY_EXCEEDED_MESSAGE in caplog.text


def test_store_outage_is_not_logged_as_quota_exhausted(monkeypatch, searches, default_image, caplog):
    use_store(monkeypatch, FakeUsage(increase_error=usage_store.UsageStoreUnavailableError("down")))

    with caplog.at_level("WARNING"):
        asyncio.run(services.add_images(["파스타"], [{"name": "파스타"}]))

    assert services.SEARCH_USAGE_LIMIT_PER_DAY_EXCEEDED_MESSAGE not in caplog.text
    assert "사용량 확인 실패" in caplog.text


def test_image_search_sends_the_key_in_a_header_not_in_the_url(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=documents("https://blog.kakaocdn.net/a.png"))

    use_search_api(monkeypatch, handler)

    asyncio.run(services.search_images("파스타"))

    url = requests[0].url
    assert requests[0].headers["authorization"] == f"KakaoAK {KAKAO_KEY}"
    assert KAKAO_KEY not in str(url)
    assert (url.scheme, url.host, url.path) == ("https", "dapi.kakao.com", "/v2/search/image")
    assert dict(url.params) == {"query": "파스타", "size": str(services.IMAGE_SEARCH_CANDIDATE_COUNT)}


def test_image_search_prefers_originals_on_the_kakao_cdn(monkeypatch):
    urls = ["https://blog.kakaocdn.net/a.png", "https://st.kakaocdn.net/b.png", "https://blog.kakaocdn.net/c.png"]
    use_search_api(monkeypatch, lambda request: httpx.Response(200, json=documents(*urls)), needed_count="2")

    assert asyncio.run(services.search_images("파스타")) == urls[:2]


@pytest.mark.parametrize(
    "blocked",
    [
        "https://postfiles.pstatic.net/a.png",
        "https://t1.daumcdn.net/a.png",
        "http://blog.kakaocdn.net/a.png",
        "https://blog.kakaocdn.net.evil.test/a.png",
        "https://evilkakaocdn.net/a.png",
        None,
    ],
)
def test_image_search_replaces_unusable_originals_with_thumbnails(monkeypatch, blocked):
    thumbnail = "https://search2.kakaocdn.net/thumb.png"
    body = documents(blocked, thumbnail=thumbnail)
    use_search_api(monkeypatch, lambda request: httpx.Response(200, json=body), needed_count="1")

    assert asyncio.run(services.search_images("파스타")) == [thumbnail]


def test_image_search_lists_usable_originals_before_thumbnails(monkeypatch):
    thumbnail = "https://search2.kakaocdn.net/thumb.png"
    body = documents("https://postfiles.pstatic.net/a.png", "https://blog.kakaocdn.net/b.png", thumbnail=thumbnail)
    use_search_api(monkeypatch, lambda request: httpx.Response(200, json=body), needed_count="2")

    assert asyncio.run(services.search_images("파스타")) == ["https://blog.kakaocdn.net/b.png", thumbnail]


@pytest.mark.parametrize(
    "body",
    [
        documents(),
        documents("https://postfiles.pstatic.net/a.png", thumbnail="http://search1.kakaocdn.net/thumb.png"),
    ],
)
def test_image_search_without_a_usable_result_returns_default_image(monkeypatch, default_image, body):
    use_search_api(monkeypatch, lambda request: httpx.Response(200, json=body))

    assert asyncio.run(services.search_images("파스타")) == [default_image]


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_image_search_error_status_returns_default_image_and_logs_the_status(monkeypatch, default_image, caplog, status):
    use_search_api(monkeypatch, lambda request: httpx.Response(status, json={"errorType": "stub"}))

    with caplog.at_level("WARNING"):
        urls = asyncio.run(services.search_images("파스타"))

    assert urls == [default_image]
    assert str(status) in caplog.text
    assert services.SEARCH_USAGE_LIMIT_PER_DAY_EXCEEDED_MESSAGE not in caplog.text


def test_image_search_timeout_returns_default_image(monkeypatch, default_image, caplog):
    def handler(request):
        raise httpx.ConnectTimeout("timeout")

    use_search_api(monkeypatch, handler)

    with caplog.at_level("WARNING"):
        urls = asyncio.run(services.search_images("파스타"))

    assert urls == [default_image]
    assert "ConnectTimeout" in caplog.text


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"meta": {}}),
        httpx.Response(200, json={"documents": [{"thumbnail_url": "https://search1.kakaocdn.net/t.png"}]}),
        httpx.Response(200, json={"documents": [{"image_url": "https://blog.kakaocdn.net/a.png"}]}),
    ],
)
def test_image_search_malformed_response_returns_default_image(monkeypatch, default_image, response):
    use_search_api(monkeypatch, lambda request: response)

    assert asyncio.run(services.search_images("파스타")) == [default_image]


def test_image_search_invalid_needed_count_returns_default_image_without_spending_quota(monkeypatch, default_image):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=documents("https://blog.kakaocdn.net/a.png"))

    use_search_api(monkeypatch, handler, needed_count="image-count")

    assert asyncio.run(services.search_images("파스타")) == [default_image]
    assert requests == []


def test_menu_keeps_its_items_when_image_search_times_out(monkeypatch, default_image):
    def handler(request):
        raise httpx.ReadTimeout("timeout")

    use_search_api(monkeypatch, handler)
    use_store(monkeypatch, FakeUsage(after=1))
    menu = [{"name": "파스타"}, {"name": "샐러드"}]

    asyncio.run(services.add_images(["파스타", "샐러드"], menu))

    assert [item["imageUrls"] for item in menu] == [[default_image], [default_image]]


def test_image_search_error_message_carrying_the_key_is_not_written_to_logs(monkeypatch, default_image, caplog):
    # 키 끝에 개행이 있으면 httpx가 헤더 값을 그대로 오류 메시지에 싣는다.
    def handler(request):
        raise httpx.LocalProtocolError(f"Illegal header value b'KakaoAK {KAKAO_KEY}'")

    use_search_api(monkeypatch, handler)

    with caplog.at_level("DEBUG"):
        urls = asyncio.run(services.search_images("파스타"))

    assert urls == [default_image]
    assert "LocalProtocolError" in caplog.text
    assert KAKAO_KEY not in caplog.text


def test_generate_sync_response_adds_images_only_with_ingredients(monkeypatch, searches):
    class FakeChain:
        def invoke(self, options):
            return langchain_core.messages.AIMessage(content='[{"name": "파스타"}]')

    class FakePrompt:
        def __or__(self, llm):
            return FakeChain()

    monkeypatch.setattr(services.langchain_core.prompts, "PromptTemplate", lambda **kwargs: FakePrompt())
    monkeypatch.setattr(services.llm_picker, "get_llm", lambda llm_type: object())
    fake = FakeUsage(after=1)
    use_store(monkeypatch, fake)

    with_ingredients = asyncio.run(services.generate_sync_response("mistral", "메뉴", {"ingredients": ["토마토"]}))
    assert with_ingredients[0]["imageUrls"] == ["https://image.test/파스타.png"]
    assert fake.increased == [1]

    fake.increased.clear()
    without_ingredients = asyncio.run(services.generate_sync_response("mistral", "메뉴", {"재료": ["토마토"]}))
    assert "imageUrls" not in without_ingredients[0]
    assert fake.increased == []
