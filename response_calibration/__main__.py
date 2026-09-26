"""Estimate scales from a candidate library and unlabeled Probe-A queries."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .calibration import ridge_fit_predict
from .features import FEATURES


def read_rows(path, labeled=False):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = ["case_name", *FEATURES]
        if labeled:
            required += ["E_scale", "density_scale"]
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: required columns: {required}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no rows")
    names = [r["case_name"] for r in rows]
    if len(set(names)) != len(names) or any(not n.strip() for n in names):
        raise ValueError(f"{path}: case names must be nonempty and unique")
    for row in rows:
        for key in required[1:]:
            if not np.isfinite(float(row[key])):
                raise ValueError(f"{path}: non-finite {key}")
        if labeled and min(float(row[k]) for k in ("E_scale", "density_scale")) <= 0:
            raise ValueError(f"{path}: material scales must be positive")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in (args.library.resolve(), args.queries.resolve()):
        parser.error("output must not overwrite an input")
    library = read_rows(args.library, labeled=True)
    queries = read_rows(args.queries)
    if set(r["case_name"] for r in library) & set(r["case_name"] for r in queries):
        parser.error("library and query case names must be disjoint")
    predictions = []
    for query in queries:
        e, rho, details = ridge_fit_predict(library, query, alpha=1e-3, local_k=10)
        predictions.append({"case_name": query["case_name"], "E_scale": e,
                            "density_scale": rho, "details": details})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"method": "local_ridge", "alpha": 1e-3,
                                      "k": 10, "predictions": predictions},
                                     indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(predictions)} estimates to {args.output}")


if __name__ == "__main__":
    main()
