"""Candidate-only standardization, response retrieval and ridge estimation.

Extracted from the experiment implementation; no simulator dependency.
"""
import numpy as np
from .features import FEATURES, EPS

def to_float(value):
    return float(value)

def candidate_stats(rows):
    means = {}
    stds = {}
    for feature in FEATURES:
        values = np.asarray([to_float(r[feature]) for r in rows], dtype=np.float64)
        means[feature] = float(values.mean())
        std = float(values.std())
        stds[feature] = std if std > EPS else 1.0
    return means, stds

def zvector(row, means, stds):
    return np.asarray(
        [(to_float(row[f]) - means[f]) / stds[f] for f in FEATURES],
        dtype=np.float64,
    )

def rank_candidates(train_rows, target):
    means, stds = candidate_stats(train_rows)
    tz = zvector(target, means, stds)
    ranked = []
    for row in train_rows:
        distance = float(np.linalg.norm(zvector(row, means, stds) - tz))
        ranked.append((distance, row))
    ranked.sort(key=lambda item: (item[0], item[1]["case_name"]))
    return ranked, means, stds

def bounds(rows):
    return {
        "e_min": min(to_float(r["E_scale"]) for r in rows),
        "e_max": max(to_float(r["E_scale"]) for r in rows),
        "d_min": min(to_float(r["density_scale"]) for r in rows),
        "d_max": max(to_float(r["density_scale"]) for r in rows),
    }

def clip_prediction(e_value, d_value, train_rows):
    b = bounds(train_rows)
    e_clipped = float(np.clip(e_value, b["e_min"], b["e_max"]))
    d_clipped = float(np.clip(d_value, b["d_min"], b["d_max"]))
    return e_clipped, d_clipped, (e_clipped != e_value or d_clipped != d_value)

def predict_nearest(train_rows, target):
    ranked, _, _ = rank_candidates(train_rows, target)
    row = ranked[0][1]
    return to_float(row["E_scale"]), to_float(row["density_scale"]), {
        "neighbors": [row["case_name"]],
        "distances": [ranked[0][0]],
        "fallback": "",
        "clipped": False,
    }

def predict_knn(train_rows, target, k, weighting):
    ranked, _, _ = rank_candidates(train_rows, target)
    top = ranked[: min(k, len(ranked))]
    distances = np.asarray([x[0] for x in top], dtype=np.float64)
    if weighting == "uniform":
        weights = np.ones(len(top), dtype=np.float64)
    else:
        weights = 1.0 / (distances + 1e-8)
    weights /= np.sum(weights)
    e_value = float(sum(to_float(row["E_scale"]) * w for w, (_, row) in zip(weights, top)))
    d_value = float(
        sum(to_float(row["density_scale"]) * w for w, (_, row) in zip(weights, top))
    )
    e_value, d_value, clipped = clip_prediction(e_value, d_value, train_rows)
    return e_value, d_value, {
        "neighbors": [row["case_name"] for _, row in top],
        "distances": distances.tolist(),
        "weights": weights.tolist(),
        "fallback": "",
        "clipped": clipped,
    }

def ridge_fit_predict(train_rows, target, alpha, local_k=None):
    ranked, means, stds = rank_candidates(train_rows, target)
    selected = ranked if local_k is None else ranked[: min(local_k, len(ranked))]
    if len(selected) < 3:
        e_value, d_value, meta = predict_knn(train_rows, target, 5, "inverse")
        meta["fallback"] = "insufficient_neighbors"
        return e_value, d_value, meta
    x_rows = []
    y_e = []
    y_d = []
    for _, row in selected:
        x_rows.append(np.concatenate(([1.0], zvector(row, means, stds))))
        y_e.append(to_float(row["E_scale"]))
        y_d.append(to_float(row["density_scale"]))
    design = np.asarray(x_rows, dtype=np.float64)
    y_e = np.asarray(y_e, dtype=np.float64)
    y_d = np.asarray(y_d, dtype=np.float64)
    query = np.concatenate(([1.0], zvector(target, means, stds)))
    reg = np.eye(design.shape[1], dtype=np.float64) * alpha
    reg[0, 0] = 0.0
    try:
        lhs = design.T @ design + reg
        beta_e = np.linalg.solve(lhs, design.T @ y_e)
        beta_d = np.linalg.solve(lhs, design.T @ y_d)
        raw_e = float(query @ beta_e)
        raw_d = float(query @ beta_d)
    except np.linalg.LinAlgError:
        e_value, d_value, meta = predict_knn(train_rows, target, 5, "inverse")
        meta["fallback"] = "linalg_error"
        return e_value, d_value, meta
    e_value, d_value, clipped = clip_prediction(raw_e, raw_d, train_rows)
    return e_value, d_value, {
        "neighbors": [row["case_name"] for _, row in selected],
        "distances": [distance for distance, _ in selected],
        "alpha": alpha,
        "local_k": local_k,
        "fallback": "",
        "clipped": clipped,
        "raw_E_scale": raw_e,
        "raw_density_scale": raw_d,
    }

def parameter_errors(target, e_value, d_value):
    target_e = to_float(target["E_scale"])
    target_d = to_float(target["density_scale"])
    e_error = abs(e_value - target_e) / max(abs(target_e), EPS) * 100.0
    d_error = abs(d_value - target_d) / max(abs(target_d), EPS) * 100.0
    return e_error, d_error, 0.5 * (e_error + d_error)
