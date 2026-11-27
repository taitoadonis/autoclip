import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def settings_client(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOCLIP_DESKTOP_MODE", "1")
    monkeypatch.setenv("AUTOCLIP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "backend.log"))

    from backend.core import desktop_config
    from backend.api.v1.settings import router

    # Desktop paths are cached at import time, before later fixtures set the environment.
    monkeypatch.setattr(
        desktop_config, "_desktop_config", desktop_config.DesktopConfig()
    )
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def saved_settings_file(tmp_path):
    saved_settings = {
        "basic": {
            "app_name": "Saved Desktop Name",
            "app_version": "0.0.1-old",
            "debug_mode": True,
            "auto_start": False,
        },
        "service": {"host": "127.0.0.2", "port": 9001, "max_memory_usage": 3072},
        "api": {
            "api_keys": {
                "dashscope": "",
                "openai": "",
                "gemini": "",
                "siliconflow": "",
                "jimeng_access": "",
                "jimeng_secret": "",
            },
            "api_model": "qwen-turbo",
            "api_max_tokens": 2048,
            "api_timeout": 45,
        },
        "processing": {
            "processing_chunk_size": 2000,
            "processing_min_score": 0.5,
            "processing_max_clips": 11,
            "processing_max_retries": 4,
        },
        "logs": {"log_level": "DEBUG", "log_retention_days": 3},
        "paths": {
            "data_directory": str(tmp_path / "saved-data"),
            "cache_directory": str(tmp_path / "saved-cache"),
            "temp_directory": str(tmp_path / "saved-temp"),
        },
    }
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(
        json.dumps(saved_settings, indent=2) + "\n", encoding="utf-8"
    )
    return settings_file, saved_settings


def test_saved_settings_return_authoritative_version(
    settings_client, saved_settings_file
):
    from backend.core import version

    _, saved_settings = saved_settings_file
    response = settings_client.get("/settings/")

    assert response.status_code == 200
    assert response.json() == {
        **saved_settings,
        "basic": {**saved_settings["basic"], "app_version": version.APP_VERSION},
    }


def test_saved_settings_follow_patched_version(
    settings_client, saved_settings_file, monkeypatch
):
    from backend.core import version

    original = settings_client.get("/settings/")
    assert original.status_code == 200
    assert original.json()["basic"]["app_version"] == version.APP_VERSION

    monkeypatch.setattr(version, "APP_VERSION", "9.9.9-test")
    response = settings_client.get("/settings/")

    assert response.status_code == 200
    assert response.json()["basic"]["app_version"] == "9.9.9-test"


def test_get_settings_does_not_rewrite_saved_file(settings_client, saved_settings_file):
    settings_file, _ = saved_settings_file
    original_contents = settings_file.read_bytes()

    response = settings_client.get("/settings/")

    assert response.status_code == 200
    assert settings_file.read_bytes() == original_contents


def test_missing_settings_return_authoritative_version(settings_client, tmp_path):
    from backend.core import version

    settings_file = tmp_path / "settings.json"
    assert not settings_file.exists()

    response = settings_client.get("/settings/")

    assert response.status_code == 200
    assert response.json()["basic"]["app_version"] == version.APP_VERSION
    assert not settings_file.exists()


def test_corrupt_settings_return_authoritative_version(settings_client, tmp_path):
    from backend.core import version

    settings_file = tmp_path / "settings.json"
    settings_file.write_text("{invalid json\n", encoding="utf-8")
    original_contents = settings_file.read_bytes()

    response = settings_client.get("/settings/")

    assert response.status_code == 200
    assert response.json()["basic"]["app_version"] == version.APP_VERSION
    assert settings_file.read_bytes() == original_contents
