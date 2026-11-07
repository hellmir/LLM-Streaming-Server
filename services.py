import asyncio
import json
import logging
import os
import urllib.parse

import dotenv
import httpx
import langchain_core.prompts

import llm_picker
import usage_store

LINE_BREAK = "\n"
TEMPLATE_HEADER = "Given the {options} I want you to create:" + LINE_BREAK
TEMPLATE_FOOTER = LINE_BREAK + "Answer in KOREAN"

logging.basicConfig(level=logging.INFO)
# httpx는 요청 주소를 쿼리 문자열까지 INFO로 남긴다
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("services_logger")


def generate_response(llm_type: str, template_content: str, options: str):
    prompt = langchain_core.prompts.PromptTemplate(
        input_variables=["options"],
        template=TEMPLATE_HEADER + template_content + TEMPLATE_FOOTER
    )
    llm = llm_picker.get_llm(llm_type.lower() if llm_type else "")  # [CHANGED] None 안전 처리
    chain = prompt | llm

    # 응답이 콘텐츠 블록 리스트로 와도 .text는 답변 텍스트만 돌려준다(사고 블록·서명 제외)
    for i, chunk in enumerate(chain.stream({"options": options}), start=1):
        log.info("chunk %s: %s", i, chunk)
        yield chunk.text


async def generate_sync_response(llm_type: str, template_content: str, options: str):
    prompt = langchain_core.prompts.PromptTemplate(
        input_variables=["options"],
        template=TEMPLATE_HEADER + template_content + TEMPLATE_FOOTER
    )
    llm = llm_picker.get_llm(llm_type.lower() if llm_type else "")  # [CHANGED] None 안전 처리
    chain = prompt | llm

    response = chain.invoke({"options": options})
    response_content = response.text
    log.info("LLM API로부터의 수신 데이터: %s", response_content)

    try:
        response_content = json.loads(response_content)
    except json.JSONDecodeError:
        log.info("JSON 파싱 실패: %s", response_content)
        return response_content

    food_names = [item["name"] for item in response_content if "name" in item]
    log.info("추천 메뉴 목록: %s", food_names)

    if isinstance(options, dict) and "ingredients" in options:  # [CHANGED] 옵션 타입/키 방어
        await add_images(food_names, response_content)

    return response_content


dotenv.load_dotenv()
KAKAO_REST_API_KEY = os.environ.get("KAKAO_REST_API_KEY")
KAKAO_IMAGE_SEARCH_URL = "https://dapi.kakao.com/v2/search/image"
DEFAULT_IMAGE_URL = os.environ.get("DEFAULT_IMAGE_URL")
SEARCH_USAGE_LIMIT_PER_DAY_EXCEEDED_MESSAGE = "일일 검색 사용량을 모두 소진하여 기본 이미지로 대체되었습니다."
# Kakao 이미지 검색의 일일 쿼터
IMAGE_SEARCH_DAILY_LIMIT = 30_000
# 쓸 수 없는 원본을 걸러도 필요한 개수가 남도록 넉넉히 받는다
IMAGE_SEARCH_CANDIDATE_COUNT = 10


async def add_images(food_names, response_content):
    image_urls = await get_image_urls(food_names)

    for i, item in enumerate(response_content):
        if i < len(image_urls):
            item["imageUrls"] = image_urls[i]

    updated_json = json.dumps(response_content, ensure_ascii=False, indent=2)
    log.info("최종 JSON 데이터: %s", updated_json)


async def get_image_urls(food_names):
    if not reserve_image_search_usage(len(food_names)):
        return [[DEFAULT_IMAGE_URL] for _ in food_names]

    return await asyncio.gather(*[search_images(name) for name in food_names])


def reserve_image_search_usage(count):
    # 동시 요청이 한도를 함께 넘지 않도록 먼저 올리고, 한도에 걸리면 되돌린다
    try:
        # 그 사이 날짜가 바뀌어도 올린 날짜의 카운터를 되돌리도록 날짜를 한 번만 구한다
        usage_day = usage_store.current_usage_day()
        image_search_usage = usage_store.increase_image_search_usage(count, usage_day)
        log.info("일일 이미지 검색 사용량: %s", image_search_usage)

        if image_search_usage < IMAGE_SEARCH_DAILY_LIMIT:
            return True

        usage_store.decrease_image_search_usage(count, usage_day)
        log.warning(SEARCH_USAGE_LIMIT_PER_DAY_EXCEEDED_MESSAGE)
        return False
    except usage_store.UsageStoreUnavailableError as e:
        log.warning("이미지 검색 사용량 확인 실패: %s", e)
        return False


async def search_images(food_name):
    # 이미지는 메뉴의 부가 정보이므로 검색이 실패해도 메뉴 응답은 기본 이미지로 내보낸다
    try:
        needed_count = int(os.environ.get("NEEDED_IMAGE_COUNT"))

        async with httpx.AsyncClient() as client:
            response = await client.get(
                KAKAO_IMAGE_SEARCH_URL,
                params={"query": food_name, "size": IMAGE_SEARCH_CANDIDATE_COUNT},
                headers={"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}
            )

        if response.status_code != 200:
            log.warning("이미지 검색 실패: 응답 코드 %s", response.status_code)
            return [DEFAULT_IMAGE_URL]

        image_urls = pick_image_urls(response.json()["documents"], needed_count)
    except (httpx.HTTPError, httpx.InvalidURL, LookupError, AttributeError, TypeError, ValueError) as e:
        # 오류 메시지에 인증 헤더 값이 실릴 수 있어 종류만 남긴다
        log.warning("이미지 검색 실패: %s", type(e).__name__)
        return [DEFAULT_IMAGE_URL]

    for image_url in image_urls:
        log.info("이미지 주소: %s", image_url)

    return image_urls or [DEFAULT_IMAGE_URL]


def pick_image_urls(documents, count):
    # 원본은 외부 참조를 막는 사이트가 많아 Kakao CDN만 쓴다(2026-11-08 실측: postfiles.pstatic.net 0/11, blog.kakaocdn.net 17/17)
    # 썸네일은 해상도가 낮지만 전부 Kakao CDN이다(실측 50/50)
    originals = [document["image_url"] for document in documents]
    thumbnails = [document["thumbnail_url"] for document in documents]
    return [url for url in originals + thumbnails if is_kakao_cdn(url)][:count]


def is_kakao_cdn(url):
    parsed = urllib.parse.urlparse(url)
    return parsed.scheme == "https" and (parsed.hostname or "").endswith(".kakaocdn.net")
