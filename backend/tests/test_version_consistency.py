import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool


BACKEND_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def version_app_factory(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOCLIP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AUTOCLIP_APP_DIR", str(tmp_path / "app"))
    monkeypatch.setenv("AUTOCLIP_MODE", "web")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "backend.log"))

    from backend.core.config import settings

    monkeypatch.setattr(settings, "log_file", str(tmp_path / "backend.log"))

    from backend import app_factory

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    monkeypatch.setattr(app_factory, "engine", engine)
    try:
        yield app_factory.create_app
    finally:
        engine.dispose()


def test_backend_version_literal_has_one_source():
    locations = []
    for path in sorted(BACKEND_ROOT.rglob("*.py")):
        relative_path = path.relative_to(BACKEND_ROOT)
        if "tests" in relative_path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        locations.extend(
            relative_path.as_posix()
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and node.value == "1.0.0"
        )

    assert locations == ["core/version.py"]


@pytest.mark.parametrize("mode", ["web", "desktop"])
def test_http_versions_match_source(version_app_factory, mode):
    from backend.core import version

    with TestClient(version_app_factory(mode)) as client:
        root_health = client.get("/health")
        api_health = client.get("/api/health/")
        openapi = client.get("/openapi.json")

    assert root_health.status_code == 200
    assert api_health.status_code == 200
    assert openapi.status_code == 200
    assert root_health.json() == {
        "status": "ok",
        "mode": mode,
        "version": version.APP_VERSION,
    }
    assert api_health.json()["status"] == "healthy"
    assert api_health.json()["version"] == version.APP_VERSION
    assert openapi.json()["info"]["version"] == version.APP_VERSION


@pytest.mark.parametrize("mode", ["web", "desktop"])
@pytest.mark.parametrize("path", ["/health", "/api/health/", "/openapi.json"])
def test_http_versions_follow_patched_source(
    version_app_factory, monkeypatch, mode, path
):
    from backend.core import version

    with TestClient(version_app_factory(mode)) as client:
        original = client.get(path)
    assert original.status_code == 200
    original_body = original.json()
    assert (
        original_body["info"]["version"]
        if path == "/openapi.json"
        else original_body["version"]
    ) == version.APP_VERSION

    monkeypatch.setattr(version, "APP_VERSION", "9.9.9-test")

    with TestClient(version_app_factory(mode)) as client:
        response = client.get(path)
    assert response.status_code == 200
    body = response.json()
    assert (
        body["info"]["version"] if path == "/openapi.json" else body["version"]
    ) == "9.9.9-test"


@pytest.mark.usefixtures("version_app_factory")
def test_desktop_config_follows_patched_source(monkeypatch):
    from backend.core import version
    from backend.core.desktop_config import get_desktop_config

    config = get_desktop_config()
    assert config.app_version == version.APP_VERSION
    monkeypatch.setattr(version, "APP_VERSION", "9.9.9-test")
    assert config.app_version == "9.9.9-test"
    assert config.dict()["app_version"] == "9.9.9-test"


@pytest.mark.usefixtures("version_app_factory")
def test_settings_default_follows_patched_source(monkeypatch):
    from backend.api.v1.settings import BasicSettings, DesktopSettings
    from backend.core import version

    assert BasicSettings().app_version == version.APP_VERSION
    monkeypatch.setattr(version, "APP_VERSION", "9.9.9-test")
    assert BasicSettings().app_version == "9.9.9-test"
    assert DesktopSettings().basic.app_version == "9.9.9-test"
    assert BasicSettings(app_version="custom-version").app_version == "custom-version"


def test_version_import_has_no_application_side_effects(tmp_path):
    script = """
import logging
import os
import sys
from pathlib import Path

before_files = set(Path.cwd().rglob("*"))
root_logger = logging.getLogger()
before_logging = (
    root_logger.level,
    tuple(root_logger.handlers),
    dict(root_logger.manager.loggerDict),
)

def reject_side_effects(event, args):
    if event.startswith("socket.") or event in {
        "os.mkdir", "os.remove", "os.rmdir", "os.rename", "os.link", "os.symlink"
    }:
        raise AssertionError(f"Application side effect: {event}")
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
    if event == "open" and args[2] & write_flags:
        raise AssertionError(f"Application file write: {args[0]}")

sys.addaudithook(reject_side_effects)
from backend.core import version

assert version.APP_VERSION == "1.0.0"
assert set(Path.cwd().rglob("*")) == before_files
assert (
    root_logger.level,
    tuple(root_logger.handlers),
    dict(root_logger.manager.loggerDict),
) == before_logging
assert not {
    "backend.core.config", "backend.core.desktop_config", "backend.core.database"
}.intersection(sys.modules)
"""
    environment = {
        **os.environ,
        "PYTHONPATH": str(BACKEND_ROOT.parent),
        "HOME": str(tmp_path),
        "AUTOCLIP_DATA_DIR": str(tmp_path / "data"),
        "AUTOCLIP_APP_DIR": str(tmp_path / "app"),
        "LOG_FILE": str(tmp_path / "backend.log"),
    }
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert list(tmp_path.iterdir()) == []
