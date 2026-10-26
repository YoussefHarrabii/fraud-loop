"""Download the data bundle into the image at build time and verify its checksum.

The bundle is a tar.gz holding `data/app.db` and `artifacts/final`. It lives as a
release asset because it is far too large for git. Nothing secret is in it.
"""

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

BLOCK = 1 << 20


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(BLOCK), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, target: Path) -> None:
    with urllib.request.urlopen(url, timeout=120) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out, BLOCK)


def extract(archive: Path, dest: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        try:
            tar.extractall(dest, filter="data")
        except TypeError:  # Python < 3.11.4 has no filter argument
            tar.extractall(dest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/data.json")
    parser.add_argument("--dest", default=".")
    parser.add_argument("--url", default=None)
    parser.add_argument("--sha256", default=None)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    url = args.url or manifest.get("url")
    expected = args.sha256 or manifest.get("sha256")
    if not url:
        print("data manifest has no url; run scripts/publish_data.py", file=sys.stderr)
        return 2

    archive = Path("/tmp/fraud-data.tar.gz") if Path("/tmp").is_dir() else Path("fraud-data.tar.gz")
    print(f"downloading {url}")
    download(url, archive)

    actual = digest(archive)
    if expected and actual != expected:
        print(f"checksum mismatch: expected {expected}, got {actual}", file=sys.stderr)
        return 3

    print(f"checksum ok {actual}")
    extract(archive, Path(args.dest))
    archive.unlink(missing_ok=True)
    print("data bundle unpacked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
