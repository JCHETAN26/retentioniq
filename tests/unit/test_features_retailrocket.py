import pandas as pd
import pytest

from retentioniq.features.retailrocket import (
    ALL_FEATURES,
    SnapshotConfig,
    build_snapshot,
    derive_cutoffs,
    sessionize,
)

CUTOFF = pd.Timestamp("2015-07-01")


def _ev(visitor: int, ts: str, kind: str = "view", item: int = 1, cat: float = 10.0) -> dict:  # type: ignore[type-arg]
    return {
        "visitor_id": visitor,
        "event_ts": pd.Timestamp(ts),
        "event_type": kind,
        "item_id": item,
        "category_id": cat,
    }


def _events() -> pd.DataFrame:
    rows = [
        # visitor 1: two sessions before cutoff, returns after -> active
        _ev(1, "2015-06-10 10:00"),
        _ev(1, "2015-06-10 10:05", "addtocart"),
        _ev(1, "2015-06-20 09:00", item=2, cat=11.0),
        _ev(1, "2015-06-20 09:10", "transaction"),
        _ev(1, "2015-07-05 12:00"),
        # visitor 2: two sessions, never returns -> inactive
        _ev(2, "2015-06-15 08:00"),
        _ev(2, "2015-06-25 08:00"),
        # visitor 3: one session only -> not eligible
        _ev(3, "2015-06-28 08:00"),
        # visitor 4: sessions only before the observation window -> not eligible
        _ev(4, "2015-04-01 08:00"),
        _ev(4, "2015-04-10 08:00"),
    ]
    return sessionize(pd.DataFrame(rows), 30)


def test_eligibility_and_labels() -> None:
    snap = build_snapshot(_events(), CUTOFF).set_index("visitor_id")
    assert sorted(snap.index) == [1, 2]
    assert snap.loc[1, "inactive"] == 0 and snap.loc[2, "inactive"] == 1
    assert snap.loc[1, "purchase_inactive"] == 1


def test_feature_values() -> None:
    s = build_snapshot(_events(), CUTOFF).set_index("visitor_id").loc[1]
    assert s["sessions_obs"] == 2
    assert s["transactions_obs"] == 1
    assert s["recency_days"] == pytest.approx(
        (CUTOFF - pd.Timestamp("2015-06-20 09:10")).days + 14 / 24 + 50 / 1440
    )
    assert s["cart_rate"] == pytest.approx(1 / 2)
    assert s["unique_categories_obs"] == 2
    assert s["ever_purchased"] == 1


def test_no_future_leakage() -> None:
    """Adding arbitrary future events must not change any feature."""
    base = build_snapshot(_events(), CUTOFF)
    extra = pd.DataFrame(
        [_ev(2, "2015-07-02 10:00", "transaction"), _ev(1, "2015-07-20 10:00", "addtocart")]
    )
    ev = sessionize(pd.concat([_events().drop(columns="session_id"), extra]), 30)
    leaked = build_snapshot(ev, CUTOFF)
    pd.testing.assert_frame_equal(base[ALL_FEATURES], leaked[ALL_FEATURES])
    assert leaked.set_index("visitor_id").loc[2, "inactive"] == 0


def test_no_missing_features() -> None:
    snap = build_snapshot(_events(), CUTOFF)
    assert not snap[ALL_FEATURES].isna().any().any()


def test_cutoffs_respect_label_observability() -> None:
    cfg = SnapshotConfig()
    c = derive_cutoffs(pd.Timestamp("2015-05-03"), pd.Timestamp("2015-09-18 02:59"), cfg)
    h = pd.Timedelta(days=cfg.horizon_days)
    assert c["test"] == [pd.Timestamp("2015-08-19")]
    assert max(c["train"]) + h <= c["validation"][0]
    assert c["validation"][0] + h <= c["test"][0]
    assert min(c["train"]) - pd.Timedelta(days=cfg.observation_days) >= pd.Timestamp("2015-05-03")
