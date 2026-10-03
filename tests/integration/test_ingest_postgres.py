"""Idempotency and quarantine behaviour against a real Postgres (`make up`)."""

from pathlib import Path

import psycopg
import pytest

from retentioniq.config import settings
from retentioniq.ingestion.load import (
    RejectRateExceededError,
    ensure_tables,
    ingest_dataset,
    libpq_url,
)
from retentioniq.ingestion.sources import RETAILROCKET, Dataset

pytestmark = pytest.mark.integration

HEADER = "timestamp,visitorid,event,itemid,transactionid\n"
GOOD = "1433221332117,1,view,10,\n1433221332118,1,transaction,10,7\n"
BAD = "1433221332119,x,view,10,\n"

DATASET = Dataset(
    name="it-retailrocket",
    url="unused",
    archive=None,
    files=(RETAILROCKET.file("events.csv"),),
)


TEST_DB = "retentioniq_test"


@pytest.fixture
def test_url() -> str:
    """A throwaway database, so tests can never touch the real warehouse's raw tables."""
    base = settings.database_url
    try:
        admin = psycopg.connect(libpq_url(base), autocommit=True)
    except psycopg.OperationalError:
        pytest.skip("Postgres not available; run `make up`")
    with admin:
        exists = admin.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,)
        ).fetchone()
        if not exists:
            admin.execute(f"CREATE DATABASE {TEST_DB}")
    return base.rsplit("/", 1)[0] + f"/{TEST_DB}"


@pytest.fixture
def conn(test_url: str) -> psycopg.Connection:
    c = psycopg.connect(libpq_url(test_url), autocommit=True)
    c.execute("DROP SCHEMA IF EXISTS raw, quarantine, meta CASCADE")
    ensure_tables(c, DATASET)
    return c


def _write(raw: Path, body: str) -> None:
    d = raw / DATASET.name
    d.mkdir(parents=True, exist_ok=True)
    (d / "events.csv").write_text(HEADER + body)


def _count(conn: psycopg.Connection, q: str) -> int:
    row = conn.execute(q, (DATASET.name,) if "%s" in q else None).fetchone()
    assert row is not None
    return int(row[0])


def test_rerun_is_idempotent_and_changed_file_replaces(
    conn: psycopg.Connection, test_url: str, tmp_path: Path
) -> None:
    raw_rows = "SELECT count(*) FROM raw.retailrocket_events"
    _write(tmp_path, GOOD * 100 + BAD)

    (r1,) = ingest_dataset(test_url, DATASET, tmp_path)
    assert (r1.status, r1.loaded_count, r1.rejected_count) == ("loaded", 200, 1)
    (r2,) = ingest_dataset(test_url, DATASET, tmp_path)
    assert r2.status == "skipped"
    assert _count(conn, raw_rows) == 200
    q = "SELECT count(*) FROM quarantine.rejected_records WHERE dataset = %s"
    assert _count(conn, q) == 1

    _write(tmp_path, GOOD * 120)
    (r3,) = ingest_dataset(test_url, DATASET, tmp_path)
    assert (r3.status, r3.loaded_count) == ("loaded", 240)
    assert _count(conn, raw_rows) == 240
    assert _count(conn, q) == 0
    status = "SELECT count(*) FROM meta.ingestion_runs WHERE dataset = %s AND status = 'loaded'"
    assert _count(conn, status) == 1


def test_reject_rate_gate_rolls_back(
    conn: psycopg.Connection, test_url: str, tmp_path: Path
) -> None:
    _write(tmp_path, GOOD + BAD * 5)
    with pytest.raises(RejectRateExceededError):
        ingest_dataset(test_url, DATASET, tmp_path)
    assert _count(conn, "SELECT count(*) FROM meta.ingestion_runs WHERE dataset = %s") == 0
