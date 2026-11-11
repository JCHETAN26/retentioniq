"""Generate small SYNTHETIC sample parquet datasets for the portfolio notebooks.

The rows are drawn from a seeded random generator that mimics each real dataset's schema.
They are not subsampled from the real data, so anything computed on them demonstrates the
method only and must not be reported as a finding.

Saves compact, standalone sample datasets to data/sample/:
- data/sample/online_retail_sample.parquet (~3,000 customers)
- data/sample/retailrocket_sample.parquet (~8,000 visitors)
- data/sample/criteo_sample.parquet (~50,000 rows)

Total compressed size remains well under 5 MB.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from retentioniq import SEED

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"


def generate_online_retail_sample(n_customers: int = 3000) -> pd.DataFrame:
    """Generate realistic Online Retail II sample orders."""
    rng = np.random.default_rng(SEED)
    countries = ["United Kingdom", "Germany", "France", "EIRE", "Spain", "Netherlands"]
    country_weights = [0.88, 0.04, 0.03, 0.02, 0.015, 0.015]

    descriptions = [
        ("WHITE HANGING HEART T-LIGHT HOLDER", 2.55),
        ("REGENCY CAKESTAND 3 TIER", 12.75),
        ("JUMBO BAG RED RETROSPOT", 1.95),
        ("ASSORTED COLOUR BIRD ORNAMENT", 1.69),
        ("PARTY BUNTING", 4.95),
        ("LUNCH BAG RED RETROSPOT", 1.65),
        ("SET OF 3 CAKE TINS PANTRY DESIGN", 4.95),
        ("NATURAL SLATE HEART CHALKBOARD", 2.95),
        ("HEART OF WICKER SMALL", 1.45),
        ("JAM MAKING SET WITH JARS", 4.25),
    ]

    rows = []
    base_date = pd.Timestamp("2010-12-01")
    max_days = 365

    for c_idx in range(1, n_customers + 1):
        cust_id = 10000 + c_idx
        country = rng.choice(countries, p=country_weights)
        first_day = rng.integers(0, 270)
        n_orders = int(rng.geometric(p=0.45))  # Pareto-like order count
        n_orders = min(n_orders, 12)

        cur_day = first_day
        for o_num in range(1, n_orders + 1):
            invoice = f"INV-{cust_id}-{o_num:02d}"
            order_date = base_date + pd.Timedelta(days=int(cur_day), hours=int(rng.integers(8, 18)))
            n_items = int(rng.integers(1, 5))
            for _ in range(n_items):
                desc, price = descriptions[rng.integers(0, len(descriptions))]
                stock_code = f"SKU-{rng.integers(100, 999)}"
                qty = int(rng.choice([1, 2, 4, 6, 12], p=[0.4, 0.3, 0.15, 0.1, 0.05]))
                line_amt = round(qty * price, 2)
                rows.append(
                    {
                        "invoice": invoice,
                        "stock_code": stock_code,
                        "description": desc,
                        "quantity": qty,
                        "invoice_date": order_date,
                        "price": price,
                        "customer_id": cust_id,
                        "country": country,
                        "order_value": line_amt,
                    }
                )
            cur_day += rng.integers(5, 45)
            if cur_day >= max_days:
                break

    df = pd.DataFrame(rows)
    return df.sort_values("invoice_date").reset_index(drop=True)


def generate_retailrocket_sample(n_visitors: int = 8000) -> pd.DataFrame:
    """Generate realistic RetailRocket browsing event stream."""
    rng = np.random.default_rng(SEED + 1)
    base_ts = pd.Timestamp("2015-05-03 08:00:00")
    max_days = 130

    event_types = ["view", "addtocart", "transaction"]
    event_weights = [0.95, 0.04, 0.01]

    rows = []
    tx_counter = 1

    for v_idx in range(1, n_visitors + 1):
        visitor_id = 200000 + v_idx
        start_day = rng.integers(0, max_days - 35)
        is_returning = rng.uniform() < 0.22  # ~22% returning visitors
        n_sessions = rng.integers(2, 6) if is_returning else 1

        cur_ts = base_ts + pd.Timedelta(days=int(start_day), hours=int(rng.integers(0, 24)))
        for _ in range(n_sessions):
            session_events = int(rng.integers(1, 6))
            category_id = int(rng.choice([101, 102, 103, 104, 105, 106, 107, 108]))
            for _ in range(session_events):
                event_type = rng.choice(event_types, p=event_weights)
                item_id = int(rng.integers(1000, 9999))
                tx_id = tx_counter if event_type == "transaction" else None
                if tx_id:
                    tx_counter += 1
                rows.append(
                    {
                        "visitor_id": visitor_id,
                        "event_ts": cur_ts,
                        "event_type": event_type,
                        "item_id": item_id,
                        "transaction_id": tx_id,
                        "category_id": category_id,
                    }
                )
                cur_ts += pd.Timedelta(minutes=int(rng.integers(1, 15)))
            cur_ts += pd.Timedelta(days=int(rng.integers(2, 25)))

    df = pd.DataFrame(rows)
    return df.sort_values("event_ts").reset_index(drop=True)


def generate_criteo_sample(n_rows: int = 50000) -> pd.DataFrame:
    """Generate representative Criteo uplift experiment sample."""
    rng = np.random.default_rng(SEED + 2)

    # 12 normalized continuous features f0..f11
    feat_data = {
        f"f{i}": rng.normal(loc=0.0, scale=1.0, size=n_rows).astype(np.float32) for i in range(12)
    }
    df = pd.DataFrame(feat_data)

    # Randomized assignment: 85% treatment, 15% control
    treatment = (rng.uniform(size=n_rows) < 0.85).astype(np.int8)
    df["treatment"] = treatment

    # Latent response propensity: baseline visit rate ~4%
    base_logit = -3.2 + 0.3 * df["f0"] + 0.2 * df["f1"] - 0.2 * df["f2"]
    p_base = 1.0 / (1.0 + np.exp(-base_logit))

    # CATE: true uplift is concentrated in users with f0 > 0.5
    tau = np.maximum(0.0, 0.03 * (df["f0"] - 0.2) + 0.015 * df["f1"])
    p_visit = np.clip(p_base + treatment * tau, 0.0, 1.0)

    visit = (rng.uniform(size=n_rows) < p_visit).astype(np.int8)
    df["visit"] = visit

    # Conversion given visit: baseline conversion ~10% of visitors
    p_conv = np.where(visit == 1, 0.10 + treatment * 0.04 * np.clip(df["f0"], 0, 1), 0.0)
    conversion = (rng.uniform(size=n_rows) < p_conv).astype(np.int8)
    df["conversion"] = conversion

    # Exposure: only treated users with impression
    df["exposure"] = np.where(
        treatment == 1, (rng.uniform(size=n_rows) < 0.95).astype(np.int8), 0
    ).astype(np.int8)
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating demo sample datasets...")

    # 1. Online Retail II
    or_df = generate_online_retail_sample(n_customers=2500)
    or_path = OUT_DIR / "online_retail_sample.parquet"
    or_df.to_parquet(or_path, compression="snappy", index=False)
    or_mb = or_path.stat().st_size / (1024 * 1024)
    print(f"Online Retail sample: {len(or_df):,} rows -> {or_path.name} ({or_mb:.2f} MB)")

    # 2. RetailRocket
    rr_df = generate_retailrocket_sample(n_visitors=6000)
    rr_path = OUT_DIR / "retailrocket_sample.parquet"
    rr_df.to_parquet(rr_path, compression="snappy", index=False)
    rr_mb = rr_path.stat().st_size / (1024 * 1024)
    print(f"RetailRocket sample:  {len(rr_df):,} rows -> {rr_path.name} ({rr_mb:.2f} MB)")

    # 3. Criteo
    cr_df = generate_criteo_sample(n_rows=50000)
    cr_path = OUT_DIR / "criteo_sample.parquet"
    cr_df.to_parquet(cr_path, compression="snappy", index=False)
    cr_mb = cr_path.stat().st_size / (1024 * 1024)
    print(f"Criteo sample:        {len(cr_df):,} rows -> {cr_path.name} ({cr_mb:.2f} MB)")

    tot_mb = or_mb + rr_mb + cr_mb
    print(f"Total sample size: {tot_mb:.2f} MB (under 5 MB target: {tot_mb < 5.0})")


if __name__ == "__main__":
    main()
