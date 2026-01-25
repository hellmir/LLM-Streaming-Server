# stream_formatter.py
# 마크다운 친화적 SSE 스트림 정규화기
import re
from typing import Generator, Iterable, Optional

_NEWLINE = re.compile(r"\r\n?")
_INCOMPLETE_LINE = re.compile(r"^\s*(?:#{1,6}|[-*•]|\d+[.)])\s*$")

# 번호목록/불릿/헤딩/문장종결 기준으로 줄 경계 보정 (레거시)
_RX_RULES_LEGACY = [
    # 윈도/맥 개행 정규화
    (_NEWLINE, "\n"),
    # 문장 종결 후 바로 이어지는 텍스트는 줄바꿈으로 분리 (공백 없이 붙는 경우)
    (re.compile(r"([.!?])(?=[^\s\d])"), r"\1\n"),
    # 번호 목록: "1. " 앞에 줄바꿈이 없으면 삽입
    (re.compile(r"(?<!\n)(\d+\.\s)"), r"\n\1"),
    # 불릿 목록: "- " 또는 "* " 또는 "• " 앞에 줄바꿈 삽입
    (re.compile(r"(?<!\n)([-*•]\s)"), r"\n\1"),
    # 헤딩: "#", "##" ... 앞에 줄바꿈 삽입
    (re.compile(r"(?<!\n)(#{1,6}\s)"), r"\n\1"),
    # 공백 없이 붙은 목록/헤딩 마커 분리 (구두점/공백 뒤에 오는 경우)
    (re.compile(r"(?<!\n)(?<=[\s.!?:：])(\d+[.)])(?=\S)"), r"\n\1"),
    (re.compile(r"(?<!\n)(?<=[\s.!?:：])([-*•])(?=\S)"), r"\n\1"),
    (re.compile(r"(?<!\n)(?<=[\s.!?:：])(#{1,6})(?=\S)"), r"\n\1"),
    # 줄 시작 마커에 공백 보정 (예: "###제목" -> "### 제목", "1.항목" -> "1. 항목")
    (re.compile(r"(?m)^(#{1,6})(?=\S)"), r"\1 "),
    (re.compile(r"(?m)^(\d+[.)])(?=[^\s\d])"), r"\1 "),
    (re.compile(r"(?m)^([-*•])(?=[^\s-])"), r"\1 "),
    # 문장 종결 후 공백만 있을 때 줄바꿈으로 정리 (한글/영문 공통)
    (re.compile(r"([.!?])\s+(?=[^\s])"), r"\1\n"),
    # 중복 개행 3개 이상은 2개로 압축
    (re.compile(r"\n{3,}"), "\n\n"),
]

# 최소한의 개행 정규화 (모델 출력이 이미 깔끔한 경우)
_RX_RULES_MINIMAL = [
    (_NEWLINE, "\n"),
    (re.compile(r"\n{3,}"), "\n\n"),
]


def _apply_rules(text: str, rules) -> str:
    out = text
    for rx, repl in rules:
        out = rx.sub(repl, out)
    return out


def normalize_chunks_to_lines(
        chunks: Iterable[str],
        llm_type: Optional[str] = None
) -> Generator[str, None, None]:
    """
    LLM 토큰(chunks)을 받아서 마크다운 친화적 규칙을 적용하고
    '완성된 줄' 단위로 생성한다. 마지막 미완성 잔여도 반환.
    """
    rules = _RX_RULES_LEGACY
    if (llm_type or "").lower() == "mistral":
        rules = _RX_RULES_MINIMAL

    buf = ""
    for ch in chunks:
        buf += str(ch)
        buf = _apply_rules(buf, rules)
        while "\n" in buf:
            line, rest = buf.split("\n", 1)
            if _INCOMPLETE_LINE.match(line):
                # 마커 단독 라인은 다음 토큰과 합쳐서 방출
                if rest and not rest.startswith("\n"):
                    buf = line.rstrip() + " " + rest.lstrip()
                    continue
                buf = line + "\n" + rest
                break
            yield line
            buf = rest
    if buf:
        # 남은 조각(마지막 줄)
        yield buf
