import os

import httpx

# 무료 Upstash DB는 계정당 1개라 다른 프로젝트와 함께 쓸 수 있으므로 키에 서비스 접두사를 둔다
PAID_MODEL_USAGE_KEY = "llm:usage:paid_model"
IMAGE_SEARCH_USAGE_KEY = "llm:usage:image_search"
REQUEST_TIMEOUT_SECONDS = 3


class UsageStoreUnavailableError(Exception):
    pass


def increase_paid_model_usage() -> int:
    return _execute("INCR", PAID_MODEL_USAGE_KEY)


def increase_image_search_usage(count: int) -> int:
    return _execute("INCRBY", IMAGE_SEARCH_USAGE_KEY, count)


def decrease_image_search_usage(count: int) -> int:
    return _execute("DECRBY", IMAGE_SEARCH_USAGE_KEY, count)


def reset_daily_usage() -> None:
    _execute("DEL", PAID_MODEL_USAGE_KEY, IMAGE_SEARCH_USAGE_KEY)


def _execute(*command):
    url = os.environ.get("KV_REST_API_URL")
    token = os.environ.get("KV_REST_API_TOKEN")

    if not url or not token:
        raise UsageStoreUnavailableError("KV_REST_API_URL 또는 KV_REST_API_TOKEN 환경 변수가 없습니다.")

    try:
        response = httpx.post(
            url,
            headers={"Authorization": f"Bearer {token}"},
            json=list(command),
            timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        return response.json()["result"]
    except (httpx.HTTPError, httpx.InvalidURL, KeyError, TypeError, ValueError) as e:
        raise UsageStoreUnavailableError(f"{command[0]} 실패: {type(e).__name__}") from e
