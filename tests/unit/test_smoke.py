from retentioniq import SEED, SPEC_VERSION
from retentioniq.config import Settings


def test_spec_version_matches_spec_file() -> None:
    from retentioniq.config import REPO_ROOT

    spec = (REPO_ROOT / "PROJECT_SPEC.md").read_text()
    assert f"v{SPEC_VERSION}" in spec


def test_seed_is_fixed() -> None:
    assert SEED == 20261002


def test_settings_paths() -> None:
    s = Settings()
    assert s.raw_dir == s.data_dir / "raw"
