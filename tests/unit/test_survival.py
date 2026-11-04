import pandas as pd
import pytest

from retentioniq.training.survival import (
    LANDMARK_COVARIATES,
    build_durations,
    build_landmark_durations,
)


def _ev(
    v: int,
    ts: str,
    kind: str = "view",
    item: int = 1,
    cat: float = 1.0,
) -> dict[str, object]:
    return {
        "visitor_id": v,
        "event_ts": pd.Timestamp(ts),
        "event_type": kind,
        "item_id": item,
        "category_id": cat,
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


def test_landmark_eligibility_and_durations() -> None:
    rows = [
        # v1: event at day 5 (< 7d) -> should be excluded from day-7 landmark
        _ev(1, "2015-05-01 10:00"),
        _ev(1, "2015-05-06 10:00", "transaction"),
        # v2: active every 10 days until near end -> eligible, event=0, duration = (T - 7)
        *[_ev(2, f"2015-05-{d:02d} 09:00") for d in (2, 12, 22)],
        *[_ev(2, f"2015-06-{d:02d} 09:00") for d in (1, 11, 21)],
        *[_ev(2, f"2015-07-{d:02d} 09:00") for d in (1, 11, 21, 31)],
        # v5: sessions at day 0, day 2 (cart), day 4 (purchase), day 6, day 15
        # Event at day 15 (duration 15d). Landmark duration = 15 - 7 = 8 days, event=1
        _ev(5, "2015-05-01 08:00", "view", item=101, cat=1.0),
        _ev(5, "2015-05-03 08:00", "addtocart", item=102, cat=2.0),
        _ev(5, "2015-05-05 08:00", "transaction", item=103, cat=3.0),
        _ev(5, "2015-05-07 08:00", "view", item=104, cat=4.0),
        _ev(5, "2015-05-16 08:00", "view", item=105, cat=5.0),
        # Defining T_end
        _ev(9, "2015-08-01 00:00"),
        _ev(9, "2015-08-02 00:00"),
    ]
    df_events = pd.DataFrame(rows)
    lm = build_landmark_durations(df_events, landmark_days=7).set_index("visitor_id")

    # v1 had event before day 7 -> excluded
    assert 1 not in lm.index
    # v2 and v5 are eligible
    assert 2 in lm.index
    assert 5 in lm.index

    # v5 checks
    row5 = lm.loc[5]
    assert row5["event"] == 1
    assert row5["duration_days_lm"] == pytest.approx(15.0 - 7.0, abs=0.01)
    assert row5["sessions_7d"] == 4
    assert row5["active_days_7d"] == 4
    assert row5["carted_7d"] == 1
    assert row5["purchased_7d"] == 1
    assert row5["activity_tier"] == "4+ sessions"


def test_landmark_zero_leakage_past_tlm() -> None:
    """Verifies that adding or modifying events strictly AFTER t_LM (day 7) has zero effect
    on landmark covariates measured in [origin, origin + 7d).
    """
    base_rows = [
        _ev(5, "2015-05-01 08:00", "view", item=101, cat=1.0),
        _ev(5, "2015-05-03 08:00", "addtocart", item=102, cat=2.0),
        _ev(5, "2015-05-05 08:00", "transaction", item=103, cat=3.0),
        _ev(5, "2015-05-07 08:00", "view", item=104, cat=4.0),
        _ev(5, "2015-05-20 08:00", "view", item=105, cat=5.0),
        _ev(9, "2015-08-01 00:00"),
        _ev(9, "2015-08-02 00:00"),
    ]
    df_base = pd.DataFrame(base_rows)
    lm_base = build_landmark_durations(df_base, landmark_days=7).set_index("visitor_id").loc[5]

    # Add extra activities past day 7 (e.g. day 10, day 12): new items, carts, categories
    leaked_rows = list(base_rows)
    leaked_rows.extend(
        [
            _ev(5, "2015-05-10 12:00", "addtocart", item=999, cat=99.0),
            _ev(5, "2015-05-12 12:00", "transaction", item=998, cat=98.0),
            _ev(5, "2015-05-14 12:00", "view", item=997, cat=97.0),
        ]
    )
    df_leaked = pd.DataFrame(leaked_rows)
    lm_leaked = build_landmark_durations(df_leaked, landmark_days=7).set_index("visitor_id").loc[5]

    # Verify every landmark covariate is bit-for-bit identical
    for col in LANDMARK_COVARIATES:
        assert lm_base[col] == pytest.approx(lm_leaked[col]), (
            f"Leakage detected in {col}: base={lm_base[col]} vs leaked={lm_leaked[col]}"
        )
