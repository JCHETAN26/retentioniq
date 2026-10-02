"""Load validated source files into Postgres `raw.*` tables.

Each source file is loaded in a single transaction:
- if the ledger already has a `loaded` run with the same checksum, the file is skipped;
- otherwise rows from any earlier run of the same file are replaced, valid rows are
  COPYed into `raw.<table>`, rejected rows go to `quarantine.rejected_records`, and a new
  ledger row is written to `meta.ingestion_runs`.
If the reject rate exceeds the file's `max_reject_rate`, the transaction is rolled back.
Re-running ingestion is therefore idempotent and never duplicates data.
"""

import io
import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import psycopg
from psycopg import sql

from retentioniq.ingestion.download import sha256_file
from retentioniq.ingestion.readers import reader_for
from retentioniq.ingestion.sources import METADATA_COLUMNS, Dataset, SourceFile
from retentioniq.quality.contracts import CONTRACTS, validate

NULL = r"\N"

DDL = """
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS quarantine;
CREATE SCHEMA IF NOT EXISTS meta;
CREATE TABLE IF NOT EXISTS meta.ingestion_runs (
    run_id          uuid PRIMARY KEY,
    dataset         text        NOT NULL,
    source_file     text        NOT NULL,
    target_table    text        NOT NULL,
    checksum_sha256 text        NOT NULL,
    schema_version  text        NOT NULL,
    row_count       bigint      NOT NULL,
    loaded_count    bigint      NOT NULL,
    rejected_count  bigint      NOT NULL,
    status          text        NOT NULL CHECK (status IN ('loaded', 'superseded')),
    started_at      timestamptz NOT NULL,
    finished_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ingestion_runs_file_idx
    ON meta.ingestion_runs (dataset, source_file, status);
CREATE TABLE IF NOT EXISTS quarantine.rejected_records (
    run_id      uuid   NOT NULL,
    dataset     text   NOT NULL,
    table_name  text   NOT NULL,
    source_file text   NOT NULL,
    source_row  bigint NOT NULL,
    reasons     text[] NOT NULL,
    record      jsonb  NOT NULL
);
CREATE INDEX IF NOT EXISTS rejected_records_file_idx
    ON quarantine.rejected_records (table_name, source_file);
"""


@dataclass(frozen=True)
class IngestionResult:
    dataset: str
    source_file: str
    status: str  # "loaded" | "skipped"
    run_id: str | None = None
    row_count: int = 0
    loaded_count: int = 0
    rejected_count: int = 0


class RejectRateExceededError(RuntimeError):
    pass


def libpq_url(sqlalchemy_url: str) -> str:
    return sqlalchemy_url.replace("postgresql+psycopg://", "postgresql://", 1)


def ensure_tables(conn: psycopg.Connection, dataset: Dataset) -> None:
    conn.execute(DDL)
    for f in dataset.files:
        cols = {**f.columns, **METADATA_COLUMNS}
        conn.execute(
            sql.SQL("CREATE TABLE IF NOT EXISTS raw.{} ({})").format(
                sql.Identifier(f.table),
                sql.SQL(", ").join(
                    sql.SQL("{} {}").format(sql.Identifier(c), sql.SQL(t)) for c, t in cols.items()
                ),
            )
        )
        conn.execute(
            sql.SQL("CREATE INDEX IF NOT EXISTS {} ON raw.{} (_source_file)").format(
                sql.Identifier(f"{f.table}_source_file_idx"), sql.Identifier(f.table)
            )
        )


def _copy_valid(cur: psycopg.Cursor, f: SourceFile, df: pd.DataFrame) -> None:
    cols = [*f.columns, *METADATA_COLUMNS]
    buf = io.StringIO()
    df[cols].to_csv(buf, index=False, header=False, na_rep=NULL)
    stmt = sql.SQL("COPY raw.{} ({}) FROM STDIN WITH (FORMAT csv, NULL {})").format(
        sql.Identifier(f.table), sql.SQL(", ").join(map(sql.Identifier, cols)), sql.Literal(NULL)
    )
    with cur.copy(stmt) as copy:
        copy.write(buf.getvalue())


def _insert_rejected(
    cur: psycopg.Cursor, run_id: str, dataset: str, f: SourceFile, rejected: pd.DataFrame
) -> None:
    rows = [
        (
            run_id,
            dataset,
            f.table,
            f.filename,
            int(r["_source_row"]),
            r["_reasons"],
            json.dumps({k: v for k, v in r.items() if not k.startswith("_")}, default=str),
        )
        for r in rejected.to_dict("records")
    ]
    cur.executemany(
        "INSERT INTO quarantine.rejected_records "
        "(run_id, dataset, table_name, source_file, source_row, reasons, record) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
        rows,
    )


def ingest_file(
    conn: psycopg.Connection, dataset: Dataset, f: SourceFile, path: Path, force: bool = False
) -> IngestionResult:
    checksum = sha256_file(path)
    with conn.transaction(), conn.cursor() as cur:
        # Serialize concurrent ingestion of the same file.
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"{dataset.name}/{f.filename}",))
        cur.execute(
            "SELECT checksum_sha256 FROM meta.ingestion_runs "
            "WHERE dataset = %s AND source_file = %s AND status = 'loaded'",
            (dataset.name, f.filename),
        )
        previous = cur.fetchone()
        if previous is not None and previous[0] == checksum and not force:
            return IngestionResult(dataset.name, f.filename, "skipped")

        cur.execute(
            sql.SQL("DELETE FROM raw.{} WHERE _source_file = %s").format(sql.Identifier(f.table)),
            (f.filename,),
        )
        cur.execute(
            "DELETE FROM quarantine.rejected_records WHERE table_name = %s AND source_file = %s",
            (f.table, f.filename),
        )
        cur.execute(
            "UPDATE meta.ingestion_runs SET status = 'superseded' "
            "WHERE dataset = %s AND source_file = %s AND status = 'loaded'",
            (dataset.name, f.filename),
        )
        cur.execute("SELECT now()")
        started_at = cur.fetchone()[0]  # type: ignore[index]

        run_id = str(uuid.uuid4())
        contract = CONTRACTS[f.table]
        offset = total = loaded = rejected_n = 0
        for chunk in reader_for(f.table)(path):
            n = len(chunk)
            source_rows = pd.RangeIndex(offset + 1, offset + n + 1)  # 1-based, excl. header
            offset += n
            total += n
            result = validate(chunk, contract)
            valid = result.valid.assign(
                _run_id=run_id, _source_file=f.filename, _source_row=source_rows[result.valid.index]
            )
            rejected = result.rejected.assign(_source_row=source_rows[result.rejected.index])
            _copy_valid(cur, f, valid)
            if len(rejected):
                _insert_rejected(cur, run_id, dataset.name, f, rejected)
            loaded += len(valid)
            rejected_n += len(rejected)

        if total and rejected_n / total > f.max_reject_rate:
            raise RejectRateExceededError(
                f"{f.filename}: {rejected_n}/{total} rows rejected "
                f"({rejected_n / total:.2%} > {f.max_reject_rate:.2%}); load rolled back"
            )

        cur.execute(
            "INSERT INTO meta.ingestion_runs (run_id, dataset, source_file, target_table, "
            "checksum_sha256, schema_version, row_count, loaded_count, rejected_count, status, "
            "started_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'loaded', %s)",
            (
                run_id,
                dataset.name,
                f.filename,
                f.table,
                checksum,
                f.schema_version,
                total,
                loaded,
                rejected_n,
                started_at,
            ),
        )
    return IngestionResult(dataset.name, f.filename, "loaded", run_id, total, loaded, rejected_n)


def ingest_dataset(
    database_url: str, dataset: Dataset, raw_dir: Path, force: bool = False
) -> list[IngestionResult]:
    dataset_dir = raw_dir / dataset.name
    with psycopg.connect(libpq_url(database_url), autocommit=True) as conn:
        with conn.transaction():
            ensure_tables(conn, dataset)
        return [
            ingest_file(conn, dataset, f, dataset_dir / f.filename, force) for f in dataset.files
        ]
