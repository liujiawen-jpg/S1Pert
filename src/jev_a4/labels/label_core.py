"""Margin-aware, approximate cell-pool mean-response reference labels.

Not DESeq2/edgeR, not a fitted NB model, not a biological-replicate analysis.
Input y is linear library-normalized count, not logNor. Welch tests compare
E[y_t] with r*E[y_c], directly testing the fold-change margin. Effects are
log2(E[y_t]/E[y_c]); their delta-method SE is descriptive, not the label test.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import stats

CLASS_ORDER = ("up", "unchanged", "down")
LABEL_NAMES = {-1: "unknown", 0: "up", 1: "unchanged", 2: "down"}
REASONS = {0: "confident_margin_call", 1: "insufficient_cells",
           2: "low_effective_sample_size", 3: "zero_or_nonestimable_mean",
           4: "degenerate_variance", 5: "insufficient_margin_evidence"}


@dataclass(frozen=True)
class Moments:
    n: int
    mean: np.ndarray
    variance: np.ndarray
    nonzero: np.ndarray

    def validate(self):
        shape = self.mean.shape
        if self.n < 2 or len(shape) != 1 or not shape[0]:
            raise ValueError("at least two cells and a nonempty gene axis required")
        if self.variance.shape != shape or self.nonzero.shape != shape:
            raise ValueError("moment axes differ")
        for x in (self.mean, self.variance, self.nonzero):
            if not np.isfinite(x).all() or (x < 0).any():
                raise ValueError("invalid moments")
        if (self.nonzero > self.n).any():
            raise ValueError("nonzero counts exceed cells")
        return self


def moments(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("finite nonnegative cells x genes required")
    return Moments(len(x), x.mean(0), x.var(0, ddof=1), (x > 0).sum(0)).validate()


def adjust(p, method="BH"):
    """One declared family, including untestable hypotheses as p=1."""
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("finite [0,1] p-vector required")
    if method not in ("BH", "BY"):
        raise ValueError("unknown multiplicity procedure")
    m = len(p)
    factor = 1. if method == "BH" else np.sum(1. / np.arange(1, m + 1))
    order = np.argsort(p, kind="stable")
    q_sorted = np.minimum.accumulate((p[order] * m * factor / np.arange(1, m + 1))[::-1])[::-1]
    result = np.empty(m)
    result[order] = np.minimum(q_sorted, 1.)
    return result


def scaled_welch(a, b, ratio):
    a.validate(); b.validate()
    if a.mean.shape != b.mean.shape or not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("aligned moments and positive ratio required")
    va, vb = a.variance / a.n, ratio**2 * b.variance / b.n
    se2 = va + vb
    df_den = va**2 / (a.n - 1) + vb**2 / (b.n - 1)
    valid = (se2 > 0) & (df_den > 0)
    t = np.zeros_like(se2)
    df = np.ones_like(se2)
    np.divide(a.mean - ratio * b.mean, np.sqrt(se2), out=t, where=valid)
    np.divide(se2**2, df_den, out=df, where=valid)
    greater, less = np.ones_like(t), np.ones_like(t)
    greater[valid] = stats.t.sf(t[valid], df[valid])
    less[valid] = stats.t.cdf(t[valid], df[valid])
    return greater, less, valid


def label_pair(treated, controls, ratio=1.2, alpha=.05, min_cells=20,
               min_nonzero=10, method="BH"):
    if ratio <= 1 or not 0 < alpha < 1:
        raise ValueError("ratio>1 and alpha in (0,1) required")
    treated.validate(); controls.validate()
    pu, upper_less, valid_upper = scaled_welch(treated, controls, ratio)
    lower_greater, pd, valid_lower = scaled_welch(treated, controls, 1. / ratio)
    pe = np.maximum(lower_greater, upper_less)  # TOST intersection-union p.
    estimable = (treated.mean > 0) & (controls.mean > 0)
    valid = estimable & valid_upper & valid_lower
    p = np.stack((pu, pe, pd))
    p[:, ~valid] = 1.
    # Combined 3G family: both direction claims plus equivalence claims.
    q = adjust(p.reshape(-1), method).reshape(p.shape)
    eligible = valid & (treated.nonzero >= min_nonzero) & (controls.nonzero >= min_nonzero)
    if min(treated.n, controls.n) < min_cells:
        eligible[:] = False
    labels = np.full(len(treated.mean), -1, dtype=np.int8)
    up = eligible & (q[0] <= alpha) & (treated.mean > ratio * controls.mean)
    down = eligible & (q[2] <= alpha) & (treated.mean < controls.mean / ratio)
    equivalent = eligible & (q[1] <= alpha) & (treated.mean > controls.mean / ratio) & (treated.mean < ratio * controls.mean)
    if ((up.astype(int) + down + equivalent) > 1).any():
        raise AssertionError("contradictory category calls")
    labels[up], labels[equivalent], labels[down] = 0, 1, 2
    reasons = np.full(len(labels), 5, dtype=np.uint8)
    reasons[labels >= 0] = 0
    reasons[~(valid_upper & valid_lower)] = 4
    reasons[~estimable] = 3
    reasons[valid & ~eligible] = 2
    if min(treated.n, controls.n) < min_cells:
        reasons[:] = 1
    effect, effect_se = np.full(len(labels), np.nan), np.full(len(labels), np.nan)
    effect[estimable] = np.log2(treated.mean[estimable] / controls.mean[estimable])
    effect_se[estimable] = np.sqrt(treated.variance[estimable] / treated.n / treated.mean[estimable]**2 +
                                  controls.variance[estimable] / controls.n / controls.mean[estimable]**2) / np.log(2.)
    return {"labels": labels, "reason_codes": reasons, "eligible": eligible,
            "effect_log2fc": effect, "standard_errors_log2fc": effect_se,
            "p_up": p[0], "p_equivalent": p[1], "p_down": p[2],
            "q_up": q[0], "q_equivalent": q[1], "q_down": q[2],
            "detection_treated": treated.nonzero / treated.n,
            "detection_control": controls.nonzero / controls.n,
            "mean_treated": treated.mean, "variance_treated": treated.variance,
            "nonzero_treated": treated.nonzero,
            "regression_eligible": eligible & np.isfinite(effect) & np.isfinite(effect_se)}


def counts(labels):
    labels = np.asarray(labels)
    return {name: int((labels == code).sum()) for code, name in LABEL_NAMES.items()}
