import importlib.util
import json
import pathlib

import app as app_module
import routers

ROOT = pathlib.Path(__file__).resolve().parent.parent
VERCEL_CONFIG = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
FUNCTION_PATH = "/api/index"


def rewrite_sources():
    return {rewrite["source"] for rewrite in VERCEL_CONFIG["rewrites"]}


def app_paths():
    application = app_module.app
    documentation_paths = {
        application.docs_url,
        application.swagger_ui_oauth2_redirect_url,
        application.redoc_url,
        application.openapi_url,
    }
    return documentation_paths | {route.path for route in routers.router.routes}


def test_every_app_route_is_routed_to_the_function():
    assert app_paths() - rewrite_sources() == set()


def test_rewrites_only_list_exact_app_routes():
    assert rewrite_sources() - app_paths() == set()
    assert all(rewrite["destination"] == FUNCTION_PATH for rewrite in VERCEL_CONFIG["rewrites"])


def test_function_entrypoint_exposes_the_app():
    spec = importlib.util.spec_from_file_location("vercel_entrypoint", ROOT / "api" / "index.py")
    entrypoint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entrypoint)

    assert entrypoint.app is app_module.app


def test_framework_preset_is_disabled_so_rewrites_decide_routing():
    # 프리셋이 잡히면 모든 경로가 함수로 가서 스캐너 요청마다 함수가 깨어난다.
    assert VERCEL_CONFIG["framework"] is None


def test_static_output_directory_keeps_source_files_private():
    # 프리셋 없이 public이 없으면 저장소 루트 파일이 그대로 공개된다.
    assert VERCEL_CONFIG["outputDirectory"] == "public"
    assert [path for path in (ROOT / "public").iterdir() if path.is_file()]


def test_daily_reset_cron_calls_the_reset_handler_at_1700_kst():
    crons = VERCEL_CONFIG["crons"]
    handler_paths = [route.path for route in routers.router.routes if route.endpoint is routers.reset_daily_usage]

    assert [cron["path"] for cron in crons] == handler_paths
    assert handler_paths[0] in rewrite_sources()
    # UTC 기준이다. Hobby는 하루 1회만 허용되고 이 시간대(±59분) 안에서 실행된다.
    assert crons[0]["schedule"] == "0 8 * * *"


def test_env_file_is_excluded_from_deployment():
    ignored = (ROOT / ".vercelignore").read_text(encoding="utf-8").splitlines()

    assert ".env" in ignored


def test_tracked_env_has_no_usable_cron_secret():
    # 공개 저장소의 자리 표시자가 실제 비밀값처럼 동작하면 안 된다(빈 값이면 크론 인증을 거부한다).
    lines = (ROOT / ".env").read_text(encoding="utf-8").splitlines()

    assert "CRON_SECRET=" in lines


def test_lock_targets_python_312_without_shadowing_packages():
    lock = json.loads((ROOT / "Pipfile.lock").read_text(encoding="utf-8"))

    assert lock["_meta"]["requires"]["python_version"] == "3.12"
    # PyPI의 routers·controller는 쓰지 않으며 routers는 로컬 routers.py와 이름이 겹친다.
    for name in ("routers", "controller", "apscheduler", "langchain-ollama", "black"):
        assert name not in lock["default"]


def test_scheduler_module_is_removed():
    assert not (ROOT / "scheduler.py").exists()
