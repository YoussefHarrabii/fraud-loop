from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_images_bake_no_key():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" not in text
    assert "OPENCODE_KEY" not in text
    assert "OPENROUTER_API_KEY" not in text
    assert "COPY . ." not in text


def test_image_ships_the_model_and_the_registry():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "configs/key_registry.json" in text
    assert "artifacts/final" in text
    assert "HEALTHCHECK" in text


def test_image_takes_the_database_from_the_environment():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FRAUD_DB" not in text
    assert "DATABASE_URL" in text


def test_dockerignore_keeps_the_registry_and_the_model():
    lines = [line.strip() for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()]
    assert "!configs/key_registry.json" in lines
    assert "!artifacts/final" in lines
    assert "secrets" in lines
    assert ".env" in lines


def test_images_run_as_non_root():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "USER appuser" in text
    assert "USER root" not in text


def test_heroku_builds_the_same_image():
    text = (ROOT / "heroku.yml").read_text(encoding="utf-8")
    assert "web: Dockerfile" in text
    assert "$PORT" in text
    assert "OPENCODE_KEY" not in text


def test_no_other_host_is_configured():
    assert not (ROOT / "render.yaml").exists()
    assert not (ROOT / "Dockerfile.deploy").exists()


def test_no_data_bundle_step_remains():
    assert not (ROOT / "configs" / "data.json").exists()
    assert not (ROOT / "scripts" / "fetch_data.py").exists()
    assert not (ROOT / "scripts" / "publish_data.py").exists()


def test_ci_scans_dependencies():
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "pip-audit" in text


def test_ci_provides_postgres_to_the_tests():
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "image: postgres:16" in text
    assert text.count("image: postgres:16") >= 2
    assert "pg_isready" in text
