import pandas as pd
import pytest

from retentioniq.training.survival import build_durations


def _ev(v: int, ts: str, kind: str = "view") -> dict[str, object]:
    return {
        "visitor_id": v,
        "event_ts": pd.Timestamp(ts),
        "event_type": kind,
        "item_id": 1,
        "category_id": 1.0,
    }


def test_events_censoring_and_population() -> None:
    rows = [
        # v1: sessions on day 0 and day 5, then a 60-day gap -> event at day 5
        _ev(1, "2015-05-01 10:00"),
        _ev(1, "2015-05-06 10:00", "transaction"),
        # v2: active every 10 days until near the end -> censored at T_end - 30d
        *[_ev(2, f"2015-05-{d:02d} 09:00") for d in (2, 12, 22)],
        *[_ev(2, f"2015-06-{d:02d} 09:00") for d in (1, 11, 21)],
        *[_ev(2, f"2015-07-{d:02d} 09:00") for d in (1, 11, 21, 31)],
        # v3: single session -> not a returning visitor
        _ev(3, "2015-05-03 10:00"),
        # v4: returning but first seen within 31 days of T_end -> excluded
        _ev(4, "2015-07-20 10:00"),
        _ev(4, "2015-07-25 10:00"),
        _ev(9, "2015-08-01 00:00"),
        _ev(9, "2015-08-02 00:00"),  # defines T_end
    ]
    d = build_durations(pd.DataFrame(rows)).set_index("visitor_id")
    assert sorted(d.index) == [1, 2]
    assert d.loc[1, "event"] == 1
    assert d.loc[1, "duration_days"] == pytest.approx(5.0)
    assert d.loc[1, "first_session_type"] == "viewed_only"
    assert d.loc[2, "event"] == 0
    t_end, origin = pd.Timestamp("2015-08-02"), pd.Timestamp("2015-05-02 09:00")
    expected = (t_end - pd.Timedelta(days=30) - origin).total_seconds() / 86_400
    assert d.loc[2, "duration_days"] == pytest.approx(expected)
