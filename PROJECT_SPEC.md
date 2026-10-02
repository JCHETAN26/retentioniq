# RetentionIQ — Project Specification & Experimental Contract

Status: **v0.3 — frozen before any modeling.**
Any change to a definition, metric, split, or gate after modeling begins must be
recorded in the [Change Log](#10-change-log) with the reason and date. Results
produced under an older spec version must say which version they used.

---

## 1. Purpose

RetentionIQ is a retention-decision platform. It lets a business team:

> understand customers → discover segments → measure retention → estimate
> inactivity risk → understand when/why customers disengage → analyze
> experiments → identify who responds incrementally to treatment → compare
> targeting strategies.

**Guiding principle:** every sophisticated component is evaluated against a
simpler baseline that sits underneath it. A component that does not beat its
baseline under the metrics defined here is reported as such, not hidden.

---

## 2. Datasets and scope boundaries

| ID | Dataset | Role | Unit of analysis |
|----|---------|------|------------------|
| A | RetailRocket e-commerce events | behavior → engagement → inactivity prediction, survival | `visitor_id` |
| B | UCI Online Retail II | RFM, revenue, cohorts, customer value | `customer_id` |
| C | Criteo Uplift (v2.1) | randomized experiment → treatment effects → uplift → targeting | row (user) |

**The datasets are never joined.** They are independent case studies. No
claim may combine evidence across datasets (e.g. "customers in RFM segment X
respond to treatment") because no shared entity exists.

```
Customer Intelligence          Treatment Intelligence
├── A: RetailRocket            └── C: Criteo Uplift
└── B: Online Retail II
```

Raw files are immutable. They live in `data/raw/<dataset>/` (later
`s3://retentioniq/raw/<dataset>/`) and are never edited in place.

### 2.1 Known dataset facts that constrain the design

- **A:** ~2.76M events and ~1.4M visitors over roughly 4.5 months (May–Sep
  2015). Most visitors appear only once, so the modeling population must be
  defined explicitly (§4.1). There is no reliable price field (item properties
  are hashed), so *monetary* means a **transaction-count proxy** for A.
- **B:** Dec 2009 – Dec 2011. About 20–25% of rows have no `Customer ID`.
  Cancellations are invoices prefixed with `C`. Non-product stock codes exist
  (postage, manual adjustments, bank charges).
- **C:** ~13.9M rows, features `f0..f11` (anonymized), `treatment`, `visit`,
  `conversion`, `exposure`. About 85% of rows are treated. Conversion is rare
  (well under 1%). **`exposure` is post-treatment and must never be used as a
  feature** (it would leak the outcome mechanism).

---

## 3. Questions the system answers

Each question names the dataset, the method, and the artifact that holds the
answer. The artifacts are produced by `make evaluate` (§8).

| # | Question | Data | Method | Answer artifact |
|---|----------|------|--------|-----------------|
| Q1 | What kinds of customers exist? | B, A | RFM segments; K-Means/GMM on behavioral features | `segmentation_metrics.json` |
| Q2 | Which customer groups generate the most value? | B | Revenue/AOV/frequency share by segment; concentration (top-x% share, Gini) | `segmentation_metrics.json` |
| Q3 | How does retention differ between cohorts? | B | First-purchase-month cohorts, M0..Mn retention matrices | `cohort_metrics.csv` |
| Q4 | Which behaviors precede inactivity? | A | Feature effects (logistic coefficients, SHAP), descriptive comparisons | `churn_model_comparison.csv`, SHAP artifacts |
| Q5 | When do customers become inactive? | A (B as check) | Kaplan–Meier, Cox PH with assumption tests | `survival_analysis.csv` |
| Q6 | Can we predict future inactivity? | A | Model ladder M0–M5 (§5.2) under temporal validation | `churn_model_comparison.csv`, `calibration_metrics.json` |
| Q7 | What do behavioral features add beyond RFM? | A | Feature-family ablation (§5.3) | `feature_ablation.csv` |
| Q8 | Does the treatment causally improve the outcome? | C | Difference in means (ATE) with CI; randomization checks | `experiment_results.json` |
| Q9 | Who responds *because of* the treatment? | C | CATE estimators (S/T/X-learner, uplift trees/forest) | `uplift_results.csv` |
| Q10 | Does uplift targeting beat conventional targeting? | C | Campaign simulation at fixed budgets (§6.4) | `campaign_simulation.csv` |

---

## 4. Definitions

### 4.1 Inactivity (Dataset A)

**Cutoff date** `t_c`: the moment at which features are computed. Only events
with `timestamp < t_c` may be used for features.

| Parameter | Primary value | Sensitivity values |
|-----------|---------------|--------------------|
| Observation window | 30 days before `t_c` (tenure uses all history before `t_c`) | 21 |
| Prediction (outcome) window | 30 days after `t_c` | 21 |
| Session gap | 30 minutes without events starts a new session | 15, 60 |
| Eligible population | ≥ 2 sessions in the observation window | ≥ 1 session; ≥ 3 sessions |

**Label:** `inactive = 1` if the visitor has **zero sessions** in
`[t_c, t_c + 30d)`, otherwise `0`.

Secondary label, reported but not used for model selection:
`purchase_inactive = 1` if there are no `transaction` events in the outcome
window, among visitors who had at least one transaction in the observation
window.

**Snapshots:** cutoffs are derived from the data end `T_end` (floored to a day)
and the prediction window `h`, so that every training label is observable before
the cutoff it is evaluated at:

- test cutoff = `T_end − h`
- validation cutoff = `test − h`
- training cutoffs = `validation − h − {0, 7, 14}` days, dropping any whose
  observation window would start before the data does

One visitor may appear in more than one snapshot. Whenever we compute confidence
intervals, we treat each visitor as one group (§7.2). Models are fit on the
training snapshots only; the validation snapshot is used for tuning, early
stopping and calibration, and is never refit on, so calibration stays
out-of-sample.

**Edge censoring:** the dataset ends at a fixed date `T_end`. Any cutoff with
`t_c + 30d > T_end` is excluded from labeled data.

Sensitivity analysis: we rerun the M1 and M4 models with every sensitivity value
and report how the base rate and PR-AUC change. A conclusion counts as robust
only if the ranking of models holds under every sensitivity setting. Longer
windows (60/90-day observation, 45-day prediction) are infeasible because the
dataset spans only about 138 days (see Change Log, v0.2).

### 4.2 Survival (Dataset A)

- **Population:** returning visitors (at least 2 sessions), with origin at least
  31 days before `T_end`. Single-session visitors would all have their event at
  duration 0.
- **Origin:** the visitor's first observed session.
- **Event:** the start of the visitor's first inactivity gap of at least 30 days,
  i.e. the last session before that gap.
- **Duration:** days from origin to the event.
- **Censoring:** a gap that starts at time `s` is only observable if
  `s + 30d ≤ T_end`. Visitors with no observed gap are right-censored at
  `T_end − 30d`.
- **Covariates** for Cox are measured in the first session only (time zero), plus
  calendar covariates. This avoids immortal-time bias: covariates measured over a
  later window would partly come after the event for visitors who become
  inactive early.

### 4.3 Transaction customers (Dataset B)

- **Qualifying purchase:** an invoice with a non-null `Customer ID`, an invoice
  number that does not start with `C`, `Quantity > 0`, and `Price > 0`, after
  excluding non-product stock codes (`POST`, `DOT`, `M`, `BANK CHARGES`, `AMAZONFEE`,
  `D`, `CRUK`, `C2`, `ADJUST*`, `TEST*`). The exclusion list is stored in config
  and versioned.
- **Order:** a distinct invoice number. **Order value** = Σ quantity × price.
- **Cancellations/returns:** analyzed separately. They are not netted into
  monetary value in v0.1, and the share of revenue they represent is reported.
- **Snapshot date:** one day after the last invoice in the analysis window.
- **Recency:** days from the customer's last qualifying purchase to the snapshot date.
- **Frequency:** number of distinct qualifying invoices.
- **Monetary:** total value of qualifying orders.
- **R/F/M scores:** quintiles 1–5. Higher means better, so recency is reversed.
  Frequency is ranked before it is cut into quintiles because it has many ties.
- **Cohort:** the calendar month of the first qualifying purchase.
  **Retained at Mk** means the customer has at least one qualifying purchase in
  cohort month + k.
- **Repeat purchase rate:** the share of customers with at least 2 qualifying invoices.

RFM segment mapping (on R and the average of F and M, called FM):

| Segment | Rule |
|---------|------|
| Champions | R ≥ 4, FM ≥ 4.5 |
| Loyal | R ≥ 3, FM ≥ 3.5 (and not Champions) |
| Potential Loyalists | R ≥ 4, 2 ≤ FM < 3.5 |
| New Customers | R = 5, FM < 2 |
| At Risk | R ≤ 2, FM ≥ 3 |
| Hibernating | R = 2, FM < 3 |
| Lost | R = 1, FM < 3 |
| Needs Attention | everything else |

Every customer falls into exactly one segment. A unit test enforces this.

### 4.4 Experiment (Dataset C)

- **Unit:** row. **Treatment:** `treatment ∈ {0,1}`, randomized.
- **Primary outcome:** `visit`. It has enough positive events to estimate
  heterogeneity well.
- **Secondary outcome:** `conversion`. It is the business outcome and is reported
  for every analysis, but it is not used for model selection.
- **Allowed features:** `f0..f11` only. `exposure`, `visit`, and `conversion` are
  never used as features.
- **ATE** = E[Y | T=1] − E[Y | T=0]. **Relative lift** = ATE / E[Y | T=0].

---

## 5. Inactivity prediction contract (Dataset A)

### 5.1 Temporal validation

- Snapshots are split by cutoff date (§4.1), never randomly:
  - **train:** the training cutoffs
  - **validation:** the validation cutoff, used for hyperparameters, early stopping,
    and calibration fitting
  - **test:** the test cutoff, touched **once** per model family after
    all tuning is finished
- Every label used for fitting must be observable at the cutoff the model is
  evaluated at: `train cutoff + h ≤ validation cutoff` and
  `validation cutoff + h ≤ test cutoff`.
- The exact cutoff dates are written to `artifacts/split_manifest.json` the
  first time features are built, and that file is versioned.

### 5.2 Model ladder

| ID | Model | Features |
|----|-------|----------|
| M0 | Naive: predicted risk = training base rate, ranked by recency only | — |
| M1 | Logistic regression | RFM / basic behavior |
| M2 | Logistic regression | Full feature set |
| M3 | Random forest | Full |
| M4 | Gradient boosting (LightGBM or XGBoost) | Full |
| M5 | PyTorch MLP (a sequence model only if justified, §5.5) | Full |

### 5.3 Ablation ladder (fixed model, M4; M2 is repeated as a check)

1. RFM only
2. + transaction behavior
3. + engagement (views and sessions over 7d and 30d)
4. + funnel (cart rate, conversion, view→cart, cart→purchase)
5. + temporal (trends, gaps between sessions and purchases)
6. + product behavior (category and product diversity, repeat-category rate)

Each step reports the change in PR-AUC with a bootstrap 95% CI (§7.2).

### 5.4 Metrics

| Metric | Role |
|--------|------|
| **PR-AUC** (average precision) | **Primary**: used for model selection and promotion |
| ROC-AUC | Secondary |
| Precision@K, Recall@K, K ∈ {5%, 10%, 20%} of the scored population | Secondary: operational |
| F1 at the threshold chosen on validation to maximize F1 | Secondary |
| Brier score | Calibration (lower is better) |
| Expected Calibration Error, 10 equal-mass bins, plus a reliability diagram | Calibration |
| Lift over the base rate at K | Reported |

Every metric is reported **overall** and for these **segments**: new visitors
(first session less than 14 days before the cutoff), low-activity visitors
(bottom tercile of sessions), purchasers versus non-purchasers, and the top
category groups.

### 5.5 Rules for adding complexity

- M5, or any sequence model, is kept only if it beats M4 on validation PR-AUC
  and the bootstrap CI of the difference excludes 0. Otherwise it is still
  reported, but it is not promoted.
- Calibration (isotonic or Platt, fit on validation) is applied before serving
  if it lowers validation Brier score.

---

## 6. Uplift contract (Dataset C)

### 6.1 Splits

A random 60/20/20 train/validation/test split, stratified on
`treatment × visit`, with a fixed seed. Random splitting is valid here because
rows are i.i.d. draws from a randomized experiment. The test set is touched
once.

### 6.2 Experiment analysis (before any modeling)

- **Randomization checks:** we compare the treatment ratio against the documented
  design and compute the standardized mean difference (SMD) for each of `f0..f11`.
  Balance passes if every |SMD| < 0.1.
- **ATE:** reported for `visit` and `conversion`, with a 95% CI. The CI uses a
  Wald (Newcombe for the difference) interval, cross-checked with a bootstrap.
- **Significance:** two-proportion z-test, α = 0.05, two-sided. Visit is the
  pre-registered primary test. Conversion is secondary and is reported with a
  Holm correction across the two.
- **Heterogeneity screen:** ATE within deciles of the baseline-propensity score,
  plus an interaction test.

### 6.3 Estimators (each must beat the one above it)

1. Random targeting (baseline)
2. Response model: highest P(Y=1 | X)
3. Two-model (separate treatment and control response models), scored as their difference
4. S-learner
5. T-learner
6. X-learner, causal forest, or uplift tree, where computationally feasible
7. Neural CATE model (for example TARNet), only if there is a stated reason

### 6.4 Metrics

| Metric | Role |
|--------|------|
| **Qini coefficient** (normalized) on `visit` | **Primary**: used for selection |
| AUUC | Secondary |
| Uplift@K, K ∈ {1, 5, 10, 20, 30}% | Secondary |
| Incremental conversions@K | Business outcome |

**Incremental outcomes at K** (estimated on the randomized test set, top-K by
score):

```
inc@K = ( ȳ_T(top K) − ȳ_C(top K) ) × N_K
```

where ȳ_T and ȳ_C are the outcome rates of the treated and control rows inside
the top-K group, and N_K is the size of that group. CIs come from 1,000
bootstrap resamples of the test set.

**Campaign simulation:** reach ∈ {1, 5, 10, 20, 30}%, comparing strategies 1–6.
Any treatment costs or conversion values are labeled **"SCENARIO — not observed
data"** wherever they appear.

---

## 7. Statistical conventions

### 7.1 Reproducibility

- Global seed: `20261002`. Seeds are passed explicitly to every estimator and splitter.
- Each run logs the git SHA, spec version, dataset checksum, feature-set
  version, and split manifest to MLflow.

### 7.2 Uncertainty

- We use a nonparametric bootstrap with 1,000 resamples. For Dataset A,
  resampling is clustered by `visitor_id`.
- We report 95% percentile intervals.
- A difference counts as an "improvement" only if the CI of the paired difference
  excludes 0.

### 7.3 Anti-cherry-picking rules

- The metrics, K values, splits, and primary outcomes in this document are fixed.
- Every model trained for the comparison tables is reported, including models
  that lose.
- The test sets are evaluated once per model family. Any re-evaluation is logged
  and disclosed.

---

## 8. Reproducible evaluation

`make evaluate` regenerates every reported number from raw data:

```
artifacts/
├── segmentation_metrics.json
├── cohort_metrics.csv
├── churn_model_comparison.csv
├── feature_ablation.csv
├── calibration_metrics.json
├── survival_analysis.csv
├── experiment_results.json
├── uplift_results.csv
├── campaign_simulation.csv
└── serving_benchmark.json
```

No performance number is typed by hand into the README, dashboards, or the web
app. They all read these artifacts or MLflow.

---

## 9. Production contract

### 9.1 Promotion gates (champion/challenger)

A challenger model is registered as `Production` only if **all** of these hold:

1. Test PR-AUC ≥ champion PR-AUC, and the paired bootstrap CI of the difference
   lies entirely above −0.005 (non-inferiority).
2. Brier score ≤ champion Brier × 1.05.
3. No segment listed in §5.4 loses more than 0.03 PR-AUC compared with the champion.
4. Data validation passed for the training snapshot.
5. Inference passes a schema contract test.

### 9.2 Serving SLOs (`POST /v1/churn/predict`, single record, warm container)

| Metric | Target |
|--------|--------|
| p50 latency | ≤ 25 ms |
| p95 latency | ≤ 100 ms |
| p99 latency | ≤ 250 ms |
| Throughput | ≥ 200 req/s per task (1 vCPU / 2 GB) |
| Error rate (5xx) | < 0.1% |

Batch endpoint: score 10k records in ≤ 5 s. The benchmark tool, concurrency
level, and hardware are recorded in `serving_benchmark.json`.

### 9.3 Monitoring alert thresholds (initial)

| Signal | Warn | Alert |
|--------|------|-------|
| Feature PSI (per feature) | > 0.1 | > 0.25 |
| Prediction-distribution PSI | > 0.1 | > 0.25 |
| Missingness change vs. training | +5 pp | +15 pp |
| Data freshness | > 1 day late | > 3 days late |
| PR-AUC on labeled data (once outcomes arrive) | −0.03 vs. test | −0.05 vs. test |
| ECE | > 0.05 | > 0.10 |
| p95 latency | > 100 ms | > 250 ms |
| 5xx rate | > 0.1% | > 1% |

---

## 10. Change Log

| Version | Date | Change | Reason |
|---------|------|--------|--------|
| 0.1 | 2026-10-02 | Initial contract | — |
| 0.2 | 2026-10-02 | §4.1: observation window 60 → 30 days; sensitivity values now 21-day observation and 21-day prediction; eligibility drops the now-redundant recent-event rule; cutoffs derived from `T_end` and `h`. §5.1 split rule restated to match. | With a 60-day observation window and non-overlapping train/validation/test label windows, no training cutoff fits inside RetailRocket's ~138-day span. Changed before any model was trained. |
| 0.3 | 2026-10-02 | §4.2: survival population is returning visitors; censoring at `T_end − 30d`; Cox covariates from the first session only. | Covariates over the first 14 days with the clock starting at day 0 cause immortal-time bias; the censoring time is the last point at which a 30-day gap could still have been observed. Changed before any model was fit. |
