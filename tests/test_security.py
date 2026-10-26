import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_gitignore_covers_env_and_secrets():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in text
    assert ".env.*" in text
    assert "secrets/" in text


def test_dockerignore_excludes_secret_material():
    text = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for pattern in [".env", ".env.*", "secrets", "*.pem", "*.key"]:
        assert pattern in text


def test_dockerfile_does_not_bake_keys():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" not in text
    assert "OPENROUTER_API_KEY" not in text
    assert "COPY . ." not in text


def test_secret_files_are_not_served(client):
    for path in ["/secrets/llm.env", "/.env", "/static/../secrets/llm.env", "/static/%2e%2e/secrets/llm.env"]:
        response = client.get(path)
        assert response.status_code in {400, 404}
        assert "OPENAI" not in response.text


def test_no_secret_files_are_tracked():
    try:
        files = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return
    secret = [f for f in files if f.endswith(".env") or f.startswith("secrets/") or f == ".env"]
    assert secret == []
