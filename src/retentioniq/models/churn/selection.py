"""Champion selection for the churn ladder (PROJECT_SPEC.md §5.5).

Selection sees validation data only. The function takes no test arrays, so the test split
cannot influence which model is chosen; test metrics are computed afterwards, for reporting.
"""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

from retentioniq.models.churn.latency import LATENCY_BUDGET_MS
from retentioniq.models.churn.metrics import paired_bootstrap_diff


@dataclass(frozen=True)
class Selection:
    best_by_validation: str
    selected_champion: str
    val_pr_auc: dict[str, float]
    # Paired-bootstrap CI of PR-AUC(candidate) - PR-AUC(best) on validation, for every
    # simpler candidate examined (in ladder order) until the rule stopped.
    candidates_checked: list[dict[str, Any]]
    reasoning: str
    # Models excluded before selection because their single-record p95 exceeds the budget.
    ineligible: list[dict[str, Any]] = field(default_factory=list)
    latency_budget_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_champion(
    y_val: np.ndarray,
    val_preds: Mapping[str, np.ndarray],
    complexity_order: Sequence[str],
    groups: np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    latency_p95_ms: Mapping[str, float] | None = None,
    latency_budget_ms: float = LATENCY_BUDGET_MS,
) -> Selection:
    """Simplest model whose validation PR-AUC is not significantly below the validation best.

    0. If latencies are given, models whose single-record p95 exceeds the budget are not
       eligible (spec v0.6.3): they cannot meet the serving SLO, whatever their accuracy.
    1. best = argmax validation PR-AUC among eligible models (ties toward the simpler one).
    2. Walk the ladder from the simplest model up to `best`; pick the first candidate whose
       paired-bootstrap 95% CI of PR-AUC(candidate) - PR-AUC(best) includes zero.
    """
    ineligible: list[dict[str, Any]] = []
    if latency_p95_ms is not None:
        ineligible = [
            {"model": m, "p95_ms": latency_p95_ms[m]}
            for m in complexity_order
            if m in val_preds and latency_p95_ms[m] > latency_budget_ms
        ]
    excluded = {d["model"] for d in ineligible}
    order = [m for m in complexity_order if m in val_preds and m not in excluded]
    if not order:
        raise ValueError("no eligible ladder models to select from")
    y = np.asarray(y_val)
    scores = {m: float(average_precision_score(y, val_preds[m])) for m in val_preds}
    top = max(scores[m] for m in order)
    best = next(m for m in order if scores[m] == top)

    checked: list[dict[str, Any]] = []
    champion = best
    for cand in order[: order.index(best)]:
        lo, hi = paired_bootstrap_diff(
            y, val_preds[cand], val_preds[best], average_precision_score, groups, n_boot, seed
        )
        tied = lo <= 0.0 <= hi
        checked.append(
            {
                "model": cand,
                "delta_pr_auc": scores[cand] - scores[best],
                "ci_low": lo,
                "ci_high": hi,
                "tied_with_best": tied,
            }
        )
        if tied:
            champion = cand
            break

    if champion == best:
        reasoning = f"{best} has the highest validation PR-AUC ({scores[best]:.4f}). " + (
            f"Every simpler model is significantly worse on validation "
            f"({len(checked)} checked), so it is kept."
            if checked
            else "It is the simplest model on the ladder."
        )
    else:
        c = checked[-1]
        reasoning = (
            f"{best} has the highest validation PR-AUC ({scores[best]:.4f}). {champion} is the "
            f"simplest model whose validation PR-AUC delta CI vs {best} includes zero "
            f"([{c['ci_low']:.4f}, {c['ci_high']:.4f}]), so it is selected."
        )
    if ineligible:
        reasoning += " Not eligible (single-record p95 over the {:g} ms budget): {}.".format(
            latency_budget_ms,
            ", ".join(f"{d['model']} ({d['p95_ms']:.1f} ms)" for d in ineligible),
        )
    return Selection(
        best,
        champion,
        scores,
        checked,
        reasoning,
        ineligible,
        latency_budget_ms if latency_p95_ms is not None else None,
    )
