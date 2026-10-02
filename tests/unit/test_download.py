import io
import json
import zipfile
from pathlib import Path

import pytest

from retentioniq.ingestion import download as dl
from retentioniq.ingestion.sources import Dataset, SourceFile


def _dataset() -> Dataset:
    files = tuple(SourceFile(n, "t", "v1", {}) for n in ("a.csv", "b.csv"))
    return Dataset(name="toy", url="https://example.invalid/toy.zip", archive="zip", files=files)


def _zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("nested/a.csv", "x\n1\n")
        zf.writestr("b.csv", "y\n2\n")
        zf.writestr("README.txt", "ignored")
    return buf.getvalue()


@pytest.fixture
def fake_fetch(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def fetch(url: str, dest: Path, auth: object) -> None:
        calls.append(url)
        dest.write_bytes(_zip_bytes())

    monkeypatch.setattr(dl, "_fetch", fetch)
    return calls


def test_download_extracts_expected_files_and_writes_manifest(
    tmp_path: Path, fake_fetch: list[str]
) -> None:
    out = dl.download(_dataset(), tmp_path)
    assert sorted(p.name for p in out.iterdir()) == ["_manifest.json", "a.csv", "b.csv"]
    manifest = json.loads((out / "_manifest.json").read_text())
    assert manifest["files"]["a.csv"]["sha256"] == dl.sha256_file(out / "a.csv")
    assert not (out / "a.csv").stat().st_mode & 0o222, "raw files must be read-only"


def test_download_is_idempotent(tmp_path: Path, fake_fetch: list[str]) -> None:
    dl.download(_dataset(), tmp_path)
    dl.download(_dataset(), tmp_path)
    assert len(fake_fetch) == 1
    dl.download(_dataset(), tmp_path, force=True)
    assert len(fake_fetch) == 2


def test_tampered_file_triggers_redownload(tmp_path: Path, fake_fetch: list[str]) -> None:
    out = dl.download(_dataset(), tmp_path)
    (out / "a.csv").chmod(0o644)
    (out / "a.csv").write_text("tampered")
    assert not dl.is_complete(_dataset(), out)
    dl.download(_dataset(), tmp_path)
    assert len(fake_fetch) == 2
