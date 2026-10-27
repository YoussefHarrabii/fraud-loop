import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_deploy_image_bakes_no_key():
    for name in ("Dockerfile.deploy", "Dockerfile"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "OPENAI_API_KEY" not in text
        assert "OPENCODE_KEY" not in text
        assert "OPENROUTER_API_KEY" not in text
        assert "COPY . ." not in text


def test_deploy_image_fetches_data_with_checksum():
    text = (ROOT / "Dockerfile.deploy").read_text(encoding="utf-8")
    assert "scripts/fetch_data.py" in text
    assert "test -s /app/data/app.db" in text
    assert "HEALTHCHECK" in text


def test_dockerignore_keeps_the_fetch_script():
    lines = [line.strip() for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()]
    assert "!scripts/fetch_data.py" in lines
    assert "!configs/data.json" in lines
    assert "!configs/access_keys.json" in lines
    assert lines.index("scripts/*") < lines.index("!scripts/fetch_data.py")
    assert "secrets" in lines
    assert ".env" in lines


def test_images_ship_the_hashed_registry():
    for name in ("Dockerfile", "Dockerfile.deploy"):
        assert "configs/access_keys.json" in (ROOT / name).read_text(encoding="utf-8")


def test_render_manifest_leaves_the_key_to_the_dashboard():
    text = (ROOT / "render.yaml").read_text(encoding="utf-8")
    assert "Dockerfile.deploy" in text
    assert "healthCheckPath: /health" in text
    assert "sync: false" in text
    assert "sk-" not in text


def test_data_manifest_shape():
    manifest = json.loads((ROOT / "configs" / "data.json").read_text(encoding="utf-8"))
    assert set(manifest) >= {"url", "sha256"}
    assert len(manifest["sha256"]) == 64

def test_images_run_as_non_root():
    for name in ("Dockerfile", "Dockerfile.deploy"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "USER appuser" in text, name
        assert "USER root" not in text, name


def test_ci_scans_dependencies():
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "pip-audit" in text
