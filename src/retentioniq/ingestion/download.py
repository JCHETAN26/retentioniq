"""Download source datasets into `data/raw/<dataset>/` and record a checksum manifest.

Downloads are idempotent: if every expected file exists and matches the manifest, nothing
is fetched. Raw files are made read-only after download so they are never edited in place.
"""

import hashlib
import json
import os
import stat
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from retentioniq.ingestion.sources import Dataset

MANIFEST = "_manifest.json"
CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def read_manifest(dataset_dir: Path) -> dict[str, Any] | None:
    path = dataset_dir / MANIFEST
    if not path.exists():
        return None
    manifest: dict[str, Any] = json.loads(path.read_text())
    return manifest


def is_complete(dataset: Dataset, dataset_dir: Path) -> bool:
    """True if every expected file exists and matches the recorded checksum."""
    manifest = read_manifest(dataset_dir)
    if manifest is None:
        return False
    recorded = manifest.get("files", {})
    for f in dataset.files:
        path = dataset_dir / f.filename
        if not path.exists() or f.filename not in recorded:
            return False
        if recorded[f.filename]["sha256"] != sha256_file(path):
            return False
    return True


def _auth(dataset: Dataset) -> tuple[str, str] | None:
    if not dataset.requires_kaggle_auth:
        return None
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    return (user, key) if user and key else None


def _fetch(url: str, dest: Path, auth: tuple[str, str] | None) -> None:
    with httpx.stream("GET", url, auth=auth, follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        with dest.open("wb") as out:
            for block in r.iter_bytes(CHUNK):
                out.write(block)


def download(dataset: Dataset, raw_dir: Path, force: bool = False) -> Path:
    dataset_dir = raw_dir / dataset.name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if not force and is_complete(dataset, dataset_dir):
        return dataset_dir

    expected = {f.filename for f in dataset.files}
    with tempfile.TemporaryDirectory(dir=dataset_dir) as tmp:
        tmp_path = Path(tmp)
        payload = tmp_path / "payload"
        _fetch(dataset.url, payload, _auth(dataset))
        payload_sha = sha256_file(payload)

        if dataset.archive == "zip":
            with zipfile.ZipFile(payload) as zf:
                members = {Path(n).name: n for n in zf.namelist() if Path(n).name in expected}
                missing = expected - members.keys()
                if missing:
                    raise RuntimeError(f"{dataset.name}: archive is missing {sorted(missing)}")
                for name, member in members.items():
                    with zf.open(member) as src, (tmp_path / name).open("wb") as dst:
                        while block := src.read(CHUNK):
                            dst.write(block)
        else:
            (filename,) = expected
            payload.rename(tmp_path / filename)

        files: dict[str, dict[str, Any]] = {}
        for name in sorted(expected):
            target = dataset_dir / name
            if target.exists():
                target.chmod(stat.S_IWUSR | stat.S_IRUSR)
            (tmp_path / name).replace(target)
            target.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            files[name] = {"sha256": sha256_file(target), "bytes": target.stat().st_size}

    manifest = {
        "dataset": dataset.name,
        "source_url": dataset.url,
        "downloaded_at": datetime.now(UTC).isoformat(),
        "payload_sha256": payload_sha,
        "files": files,
    }
    (dataset_dir / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return dataset_dir
