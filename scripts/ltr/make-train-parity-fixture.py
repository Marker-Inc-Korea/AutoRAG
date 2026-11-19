"""Regenerates test/fixtures/ltr/train-parity.json: a small ranking dataset plus what Python xgboost learns from it.

The WASM trainer (src/ltr/wasm-trainer.ts) must pick the same number of rounds and produce the same margins. Run with:
  uv run --with xgboost --with numpy python scripts/ltr/make-train-parity-fixture.py
"""
import json
import pathlib

import numpy as np
import xgboost as xgb

out = pathlib.Path(__file__).resolve().parents[2] / "test" / "fixtures" / "ltr" / "train-parity.json"
rng = np.random.default_rng(11)
F = 8


def make(queries, per):
    X, y, groups = [], [], []
    for _ in range(queries):
        x = rng.normal(size=(per, F))
        x[rng.random((per, F)) < 0.15] = np.nan
        signal = np.nan_to_num(x[:, 0]) * 1.4 + np.nan_to_num(x[:, 3]) * 0.8 + rng.normal(size=per) * 0.9
        label = np.zeros(per, dtype=int)
        label[np.argmax(signal)] = 3
        X.append(x)
        y.append(label)
        groups.append(per)
    return np.vstack(X), np.concatenate(y), groups


sets = {name: make(q, 12) for name, q in (("train", 90), ("valid", 25), ("test", 25))}


def dmat(name):
    X, y, g = sets[name]
    d = xgb.DMatrix(X, y, feature_names=[f"f{i}" for i in range(F)])
    d.set_group(g)
    return d


params = dict(objective="rank:pairwise", eval_metric="ndcg@10", eta=0.05, max_depth=4, min_child_weight=5,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", nthread=1)
runs = {}
for seed in (0, 1, 2, 3):
    booster = xgb.train({**params, "seed": seed}, dmat("train"), 600, evals=[(dmat("valid"), "valid")],
                        early_stopping_rounds=50, verbose_eval=False)
    n = booster.best_iteration + 1
    runs[str(seed)] = {"rounds": n, "bestScore": booster.best_score,
                       "testMargins": booster.predict(dmat("test"), iteration_range=(0, n), output_margin=True).tolist()}


def pack(name):
    X, y, g = sets[name]
    return {"rows": [[None if np.isnan(v) else float(v) for v in r] for r in X], "labels": y.tolist(), "groups": g}


out.write_text(json.dumps({"xgboost": xgb.__version__, "params": {k: v for k, v in params.items()},
                           "train": pack("train"), "valid": pack("valid"), "test": pack("test"), "runs": runs}))
print("xgboost", xgb.__version__, {s: r["rounds"] for s, r in runs.items()})
