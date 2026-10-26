"""Pack the ignored data bundle, record its checksum, and upload it as a release asset.

The image build needs `data/app.db` and `artifacts/final`, both gitignored. This
script puts them somewhere the cloud build can fetch: a GitHub release asset.

    $env:GITHUB_TOKEN = "ghp_..."     # repo scope, contents:write
    uv run python scripts/publish_data.py --repo you/fraud-detection

It writes `configs/data.json` (committed) with the URL and SHA-256, so every
later build is reproducible and fails loudly if the asset changes underneath it.
"""

import argparse
import hashlib
import json
import os
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
ASSET = "fraud-data.tar.gz"
MANIFEST = ROOT / "configs" / "data.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pack(target: Path) -> None:
    sources = [ROOT / "data" / "app.db"]
    final = ROOT / "artifacts" / "final"
    sources += sorted(p for p in final.iterdir() if p.is_file())
    missing = [str(p) for p in sources if not p.exists()]
    if missing:
        sys.exit(f"missing data: {', '.join(missing)}")

    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, "w:gz", compresslevel=6) as tar:
        for path in sources:
            tar.add(path, arcname=str(path.relative_to(ROOT)).replace("\\", "/"))


def create_release(client: httpx.Client, repo: str, tag: str) -> dict:
    response = client.get(f"https://api.github.com/repos/{repo}/releases/tags/{tag}")
    if response.status_code == 200:
        return response.json()
    created = client.post(
        f"https://api.github.com/repos/{repo}/releases",
        json={"tag_name": tag, "name": f"Data bundle {tag}", "body": "SQLite demo database and model bundle."},
    )
    if created.status_code >= 400:
        sys.exit(f"release create failed {created.status_code}: {created.text[:300]}")
    return created.json()


def upload(client: httpx.Client, repo: str, release: dict, path: Path) -> str:
    assets = client.get(f"https://api.github.com/repos/{repo}/releases/{release['id']}/assets")
    for asset in assets.json() if assets.status_code == 200 else []:
        if asset["name"] == ASSET:
            client.delete(f"https://api.github.com/repos/{repo}/releases/assets/{asset['id']}")
    with path.open("rb") as body:
        response = client.post(
            f"https://uploads.github.com/repos/{repo}/releases/{release['id']}/assets?name={ASSET}",
            headers={"Content-Type": "application/octet-stream"},
            content=body,
            timeout=None,
        )
    if response.status_code >= 400:
        sys.exit(f"upload failed {response.status_code}: {response.text[:300]}")
    return response.json()["browser_download_url"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=os.getenv("GITHUB_REPO"), help="owner/name")
    parser.add_argument("--token", default=os.getenv("GITHUB_TOKEN"))
    parser.add_argument("--tag", default="data-v1")
    parser.add_argument("--out", default=str(ROOT / "dist" / ASSET))
    parser.add_argument("--pack-only", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    print(f"packing -> {out}")
    pack(out)
    checksum = sha256(out)
    size = out.stat().st_size
    print(f"{size / 1e6:.1f} MB sha256={checksum}")

    url = ""
    if not args.pack_only:
        if not args.repo or not args.token:
            sys.exit("need --repo owner/name and GITHUB_TOKEN (or use --pack-only)")
        headers = {"Authorization": f"Bearer {args.token}", "Accept": "application/vnd.github+json"}
        with httpx.Client(headers=headers, timeout=120) as client:
            release = create_release(client, args.repo, args.tag)
            print("uploading; a 100 MB+ asset takes a while")
            url = upload(client, args.repo, release, out)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "url": url,
        "sha256": checksum,
        "bytes": size,
        "tag": args.tag,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(ROOT)}")
    if url:
        print(f"url {url}")
        print("commit configs/data.json, then deploy")
    else:
        print("no upload: set url in configs/data.json after uploading the archive by hand")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
