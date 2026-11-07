import datetime
import os

import httpx

# 무료 Upstash DB는 계정당 1개라 다른 프로젝트와 함께 쓸 수 있으므로 키에 서비스 접두사를 둔다
PAID_MODEL_USAGE_KEY = "llm:usage:paid_model"
IMAGE_SEARCH_USAGE_KEY = "llm:usage:image_search"
# 키에 날짜를 넣어 날이 바뀌면 새 카운터로 시작한다. 경계는 Google Cloud(미국 서부) 일일 한도 초기화에 맞춘 17시 KST다
USAGE_DAY_TIMEZONE = datetime.timezone(datetime.timedelta(hours=-8))
# 지난 날짜의 키는 다시 쓰지 않으므로 공용 DB에 쌓이지 않게 만료시킨다. 하루보다 길어야 쓰는 동안 사라지지 않는다
USAGE_KEY_TTL_SECONDS = 2 * 24 * 60 * 60
REQUEST_TIMEOUT_SECONDS = 3


class UsageStoreUnavailableError(Exception):
    pass


def current_usage_day() -> str:
    return _now().astimezone(USAGE_DAY_TIMEZONE).date().isoformat()


def increase_paid_model_usage() -> int:
    return _change("INCRBY", _daily_key(PAID_MODEL_USAGE_KEY, current_usage_day()), 1)


def increase_image_search_usage(count: int, usage_day: str) -> int:
    return _change("INCRBY", _daily_key(IMAGE_SEARCH_USAGE_KEY, usage_day), count)


def decrease_image_search_usage(count: int, usage_day: str) -> int:
    return _change("DECRBY", _daily_key(IMAGE_SEARCH_USAGE_KEY, usage_day), count)


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _daily_key(key, usage_day):
    return f"{key}:{usage_day}"


def _change(command, key, count):
    url = os.environ.get("KV_REST_API_URL")
    token = os.environ.get("KV_REST_API_TOKEN")

    if not url or not token:
        raise UsageStoreUnavailableError("KV_REST_API_URL 또는 KV_REST_API_TOKEN 환경 변수가 없습니다.")

    try:
        # 값만 바뀌고 만료가 빠지는 일이 없도록 한 트랜잭션으로 보낸다
        response = httpx.post(
            f"{url}/multi-exec",
            headers={"Authorization": f"Bearer {token}"},
            json=[[command, key, count], ["EXPIRE", key, USAGE_KEY_TTL_SECONDS]],
            timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        results = [item["result"] for item in response.json()]
        return results[0]
    except (httpx.HTTPError, httpx.InvalidURL, LookupError, TypeError, ValueError) as e:
        raise UsageStoreUnavailableError(f"{command} 실패: {type(e).__name__}") from e
