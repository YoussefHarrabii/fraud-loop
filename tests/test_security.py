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

def test_schema_and_docs_are_not_public(client):
    for path in ("/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"):
        response = client.get(path)
        assert response.status_code == 404, path
        assert "X-Access-Key" not in response.text


def test_registry_is_hashes_without_plaintext():
    import json

    text = (ROOT / "configs" / "key_registry.json").read_text(encoding="utf-8")
    document = json.loads(text)
    assert document["keys"]
    assert all("key" not in entry for entry in document["keys"])
    assert "fr_" not in text

def test_no_sql_string_interpolation_in_the_store():
    """A tripwire, not proof: it catches the obvious way this goes wrong later."""
    import re

    text = (ROOT / "src" / "fraud" / "store.py").read_text(encoding="utf-8")
    for pattern in (r'execute\(\s*f"', r"execute\(\s*f'", r'read_sql_query\(\s*f', r"execute\([^)]*\.format\("):
        assert not re.search(pattern, text), pattern
