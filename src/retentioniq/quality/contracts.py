"""Raw-layer data contracts.

A contract has two stages:
1. Type coercion: string columns are parsed into their declared types. A non-empty value
   that fails to parse rejects the row with reason `<column>:unparseable`.
2. A Pandera schema, validated lazily, for nullability and domain checks.

Rows that fail either stage are split off with their reasons so they can be quarantined.
Only *structural* validity is enforced here. Business rules (e.g. customer_id present,
cancellations, non-product stock codes) belong to dbt staging, see PROJECT_SPEC.md §4.3.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import pandera.pandas as pa

ColumnType = Literal["int", "float", "str", "datetime", "binary"]

EVENT_TYPES = ("view", "addtocart", "transaction")
# Epoch-ms bounds that comfortably contain the RetailRocket collection period (2015).
RR_MIN_MS = int(pd.Timestamp("2015-01-01").timestamp() * 1000)
RR_MAX_MS = int(pd.Timestamp("2016-01-01").timestamp() * 1000)


@dataclass(frozen=True)
class Contract:
    types: dict[str, ColumnType]
    schema: pa.DataFrameSchema


@dataclass(frozen=True)
class ValidationResult:
    valid: pd.DataFrame
    rejected: pd.DataFrame  # original string values plus a `_reasons` column


def _parse(s: pd.Series, kind: ColumnType) -> pd.Series:
    blank = s.isna() | (s.astype(str).str.strip() == "")
    if kind == "str":
        return s.astype(object).where(~blank, None)
    if kind == "datetime":
        return pd.to_datetime(s.where(~blank), format="%Y-%m-%d %H:%M:%S", errors="coerce")
    num = pd.to_numeric(s.where(~blank), errors="coerce")
    if kind == "float":
        return num.astype("Float64")
    # int / binary: reject non-integral numbers such as "1.5"
    num = num.where(np.isclose(num.fillna(0) % 1, 0))
    return num.astype("Int64")


def validate(df: pd.DataFrame, contract: Contract) -> ValidationResult:
    df = df.reset_index(drop=True)
    missing = set(contract.types) - set(df.columns)
    if missing:
        raise ValueError(f"source is missing columns {sorted(missing)}")

    typed = pd.DataFrame(index=df.index)
    reasons: dict[str, pd.Series] = {}
    for col, kind in contract.types.items():
        raw = df[col]
        parsed = _parse(raw, kind)
        had_value = raw.notna() & (raw.astype(str).str.strip() != "")
        reasons[f"{col}:unparseable"] = had_value & parsed.isna()
        typed[col] = parsed

    try:
        contract.schema.validate(typed, lazy=True)
    except pa.errors.SchemaErrors as err:
        fc = err.failure_cases
        if fc["index"].isna().any():
            raise  # a table-level failure (e.g. wrong dtype) is a bug, not a bad row
        # DataFrame-level (row-wise) checks are reported once per column; collapse to "row".
        row_level = fc["schema_context"].eq("DataFrameSchema") | fc["column"].isna()
        fc = fc.assign(column=fc["column"].where(~row_level, "row"))
        for (column, check), group in fc.groupby(["column", "check"]):
            mask = pd.Series(False, index=df.index)
            mask.loc[group["index"].astype(int).unique()] = True
            reasons[f"{column}:{check}"] = mask

    flags = pd.DataFrame(reasons)
    bad = flags.any(axis=1)
    rejected = df.loc[bad].copy()
    cols = flags.columns.to_numpy()
    rejected["_reasons"] = [list(cols[row]) for row in flags.loc[bad].to_numpy(dtype=bool)]
    return ValidationResult(valid=typed.loc[~bad], rejected=rejected)


def _binary(name: str) -> pa.Column:
    return pa.Column(pd.Int64Dtype(), pa.Check.isin([0, 1]), nullable=False, name=name)


CONTRACTS: dict[str, Contract] = {
    "retailrocket_events": Contract(
        types={
            "timestamp": "int",
            "visitorid": "int",
            "event": "str",
            "itemid": "int",
            "transactionid": "int",
        },
        schema=pa.DataFrameSchema(
            {
                "timestamp": pa.Column(
                    pd.Int64Dtype(), pa.Check.in_range(RR_MIN_MS, RR_MAX_MS), nullable=False
                ),
                "visitorid": pa.Column(pd.Int64Dtype(), pa.Check.ge(0), nullable=False),
                "event": pa.Column(object, pa.Check.isin(EVENT_TYPES), nullable=False),
                "itemid": pa.Column(pd.Int64Dtype(), pa.Check.ge(0), nullable=False),
                "transactionid": pa.Column(pd.Int64Dtype(), pa.Check.ge(0), nullable=True),
            },
            checks=[
                # transactionid is present exactly when the event is a transaction
                pa.Check(
                    lambda d: ((d["event"] == "transaction") == d["transactionid"].notna()).fillna(
                        False
                    ),
                    name="transactionid_iff_transaction",
                    element_wise=False,
                )
            ],
        ),
    ),
    "retailrocket_item_properties": Contract(
        types={"timestamp": "int", "itemid": "int", "property": "str", "value": "str"},
        schema=pa.DataFrameSchema(
            {
                "timestamp": pa.Column(
                    pd.Int64Dtype(), pa.Check.in_range(RR_MIN_MS, RR_MAX_MS), nullable=False
                ),
                "itemid": pa.Column(pd.Int64Dtype(), pa.Check.ge(0), nullable=False),
                "property": pa.Column(object, nullable=False),
                "value": pa.Column(object, nullable=False),
            }
        ),
    ),
    "retailrocket_category_tree": Contract(
        types={"categoryid": "int", "parentid": "int"},
        schema=pa.DataFrameSchema(
            {
                "categoryid": pa.Column(pd.Int64Dtype(), pa.Check.ge(0), nullable=False),
                "parentid": pa.Column(pd.Int64Dtype(), pa.Check.ge(0), nullable=True),
            }
        ),
    ),
    "online_retail_transactions": Contract(
        types={
            "invoice": "str",
            "stock_code": "str",
            "description": "str",
            "quantity": "int",
            "invoice_date": "datetime",
            "price": "float",
            "customer_id": "int",
            "country": "str",
            "source_sheet": "str",
        },
        schema=pa.DataFrameSchema(
            {
                "invoice": pa.Column(
                    object, pa.Check.str_matches(r"^[A-Z]?\d{6}$"), nullable=False
                ),
                "stock_code": pa.Column(object, nullable=False),
                "description": pa.Column(object, nullable=True),
                "quantity": pa.Column(pd.Int64Dtype(), nullable=False),
                "invoice_date": pa.Column(
                    "datetime64[ns]",
                    pa.Check.in_range(pd.Timestamp("2009-12-01"), pd.Timestamp("2011-12-31")),
                    nullable=False,
                ),
                "price": pa.Column(pd.Float64Dtype(), nullable=False),
                "customer_id": pa.Column(pd.Int64Dtype(), pa.Check.gt(0), nullable=True),
                "country": pa.Column(object, nullable=True),
                "source_sheet": pa.Column(object, nullable=False),
            }
        ),
    ),
    "criteo_uplift": Contract(
        types={
            **{f"f{i}": "float" for i in range(12)},
            "treatment": "binary",
            "conversion": "binary",
            "visit": "binary",
            "exposure": "binary",
        },
        schema=pa.DataFrameSchema(
            {
                **{f"f{i}": pa.Column(pd.Float64Dtype(), nullable=False) for i in range(12)},
                **{c: _binary(c) for c in ("treatment", "conversion", "visit", "exposure")},
            }
        ),
    ),
}
