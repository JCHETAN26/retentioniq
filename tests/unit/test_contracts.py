import pandas as pd
import pytest

from retentioniq.quality.contracts import CONTRACTS, validate

TS = "1433221332117"  # 2015-06-02


def _events(rows: list[list[str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["timestamp", "visitorid", "event", "itemid", "transactionid"])


def test_events_valid_rows_pass_and_are_typed() -> None:
    df = _events([[TS, "257597", "view", "355908", ""], [TS, "1", "transaction", "2", "4000"]])
    r = validate(df, CONTRACTS["retailrocket_events"])
    assert len(r.valid) == 2 and r.rejected.empty
    assert str(r.valid["visitorid"].dtype) == "Int64"
    assert r.valid["transactionid"].isna().tolist() == [True, False]


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        ([TS, "abc", "view", "1", ""], "visitorid:unparseable"),
        ([TS, "1", "click", "1", ""], "event:isin"),
        (["1000", "1", "view", "1", ""], "timestamp:in_range"),
        ([TS, "1", "view", "1", "99"], "row:transactionid_iff_transaction"),
        ([TS, "1", "transaction", "1", ""], "row:transactionid_iff_transaction"),
        ([TS, "", "view", "1", ""], "visitorid:not_nullable"),
    ],
)
def test_events_bad_rows_are_rejected_with_reason(row: list[str], reason: str) -> None:
    df = _events([[TS, "1", "view", "1", ""], row])
    r = validate(df, CONTRACTS["retailrocket_events"])
    assert len(r.valid) == 1
    assert len(r.rejected) == 1
    reasons = r.rejected["_reasons"].iloc[0]
    assert any(x.startswith(reason) for x in reasons), reasons
    # rejected rows keep their original string values for the quarantine record
    assert r.rejected.iloc[0]["visitorid"] == row[1]


def test_online_retail_keeps_business_edge_cases() -> None:
    """Cancellations, missing customers and negative quantities are valid at the raw layer."""
    cols = ["invoice", "stock_code", "description", "quantity", "invoice_date", "price",
            "customer_id", "country", "source_sheet"]  # fmt: skip
    df = pd.DataFrame(
        [
            ["489434", "85048", "LIGHTS", "12", "2009-12-01 07:45:00", "6.95", "13085", "UK", "s"],
            ["C489449", "22087", "PAPER", "-12", "2009-12-01 10:33:00", "2.95", "", "UK", "s"],
            ["A506401", "B", "Adjust bad debt", "1", "2010-04-29 13:36:00", "-53594.36", "", "", "s"],
            ["xx", "1", "BAD", "1", "not-a-date", "1.0", "1", "UK", "s"],
        ],
        columns=cols,
    )
    r = validate(df, CONTRACTS["online_retail_transactions"])
    assert len(r.valid) == 3
    reasons = set(r.rejected["_reasons"].iloc[0])
    assert {"invoice_date:unparseable"} <= reasons
    assert any(x.startswith("invoice:str_matches") for x in reasons)


def test_criteo_binary_columns() -> None:
    base = {f"f{i}": "0.5" for i in range(12)}
    ok = {**base, "treatment": "1", "conversion": "0", "visit": "1", "exposure": "0"}
    bad = {**base, "treatment": "2", "conversion": "0", "visit": "1", "exposure": "0"}
    r = validate(pd.DataFrame([ok, bad]), CONTRACTS["criteo_uplift"])
    assert len(r.valid) == 1
    assert any(x.startswith("treatment:isin") for x in r.rejected["_reasons"].iloc[0])


def test_missing_column_is_an_error() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        validate(pd.DataFrame({"timestamp": [TS]}), CONTRACTS["retailrocket_events"])
