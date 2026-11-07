import asyncio

import pytest

import services
import usage_store


class FakeUsage:
    """이미지 검색 사용량 저장소 대역."""

    def __init__(self, after=3, increase_error=None, decrease_error=None):
        self.after = after
        self.increase_error = increase_error
        self.decrease_error = decrease_error
        self.increased = []
        self.decreased = []

    def increase(self, count):
        self.increased.append(count)
        if self.increase_error:
            raise self.increase_error
        return self.after

    def decrease(self, count):
        self.decreased.append(count)
        if self.decrease_error:
            raise self.decrease_error
        return self.after - count


@pytest.fixture
def searches(monkeypatch):
    calls = []

    async def fake_search(food_name):
        calls.append(food_name)
        return [f"https://image.test/{food_name}.png"]

    monkeypatch.setattr(services, "search_google_images", fake_search)
    return calls


@pytest.fixture
def default_image(monkeypatch):
    monkeypatch.setattr(services, "DEFAULT_IMAGE_URL", "https://image.test/default.png")
    return "https://image.test/default.png"


def use_store(monkeypatch, fake):
    monkeypatch.setattr(usage_store, "increase_image_search_usage", fake.increase)
    monkeypatch.setattr(usage_store, "decrease_image_search_usage", fake.decrease)


def test_add_images_within_limit_searches_and_keeps_reservation(monkeypatch, searches):
    fake = FakeUsage(after=99)
    use_store(monkeypatch, fake)
    menu = [{"name": "파스타"}, {"name": "샐러드"}]

    asyncio.run(services.add_images(["파스타", "샐러드"], menu))

    assert searches == ["파스타", "샐러드"]
    assert fake.increased == [2]
    assert fake.decreased == []
    assert menu[0]["imageUrls"] == ["https://image.test/파스타.png"]
    assert menu[1]["imageUrls"] == ["https://image.test/샐러드.png"]


def test_add_images_at_limit_uses_default_image_and_rolls_back(monkeypatch, searches, default_image):
    fake = FakeUsage(after=100)
    use_store(monkeypatch, fake)
    menu = [{"name": "파스타"}, {"name": "샐러드"}]

    asyncio.run(services.add_images(["파스타", "샐러드"], menu))

    assert searches == []
    assert fake.increased == [2]
    assert fake.decreased == [2]
    assert menu[0]["imageUrls"] == [default_image]
    assert menu[1]["imageUrls"] == [default_image]


def test_add_images_uses_default_image_when_store_is_down(monkeypatch, searches, default_image):
    fake = FakeUsage(increase_error=usage_store.UsageStoreUnavailableError("down"))
    use_store(monkeypatch, fake)
    menu = [{"name": "파스타"}]

    asyncio.run(services.add_images(["파스타"], menu))

    assert searches == []
    assert fake.decreased == []
    assert menu[0]["imageUrls"] == [default_image]


def test_add_images_uses_default_image_when_rollback_fails(monkeypatch, searches, default_image):
    fake = FakeUsage(after=100, decrease_error=usage_store.UsageStoreUnavailableError("down"))
    use_store(monkeypatch, fake)
    menu = [{"name": "파스타"}]

    asyncio.run(services.add_images(["파스타"], menu))

    assert searches == []
    assert menu[0]["imageUrls"] == [default_image]


def test_limit_is_logged_as_quota_exhausted(monkeypatch, searches, default_image, caplog):
    use_store(monkeypatch, FakeUsage(after=100))

    with caplog.at_level("WARNING"):
        asyncio.run(services.add_images(["파스타"], [{"name": "파스타"}]))

    assert services.SEARCH_USAGE_LIMIT_PER_DAY_EXCEEDED_MESSAGE in caplog.text


def test_store_outage_is_not_logged_as_quota_exhausted(monkeypatch, searches, default_image, caplog):
    use_store(monkeypatch, FakeUsage(increase_error=usage_store.UsageStoreUnavailableError("down")))

    with caplog.at_level("WARNING"):
        asyncio.run(services.add_images(["파스타"], [{"name": "파스타"}]))

    assert services.SEARCH_USAGE_LIMIT_PER_DAY_EXCEEDED_MESSAGE not in caplog.text
    assert "사용량 확인 실패" in caplog.text


def test_generate_sync_response_adds_images_only_with_ingredients(monkeypatch, searches):
    class FakeChain:
        def __init__(self, content):
            self.content = content

        def invoke(self, options):
            return self

    class FakePrompt:
        def __or__(self, llm):
            return FakeChain('[{"name": "파스타"}]')

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
