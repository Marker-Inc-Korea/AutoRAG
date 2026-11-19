"""Regenerates test/fixtures/ltr: a tiny XGBoost ranker plus the margins the Python library computes for fixed rows.

The TypeScript predictor (src/ltr/xgboost-model.ts) must reproduce these margins. Run with xgboost installed:
  uv run --with xgboost --with numpy python scripts/ltr/make-xgb-fixture.py
"""
import json
import pathlib

import numpy as np
import xgboost as xgb

out = pathlib.Path(__file__).resolve().parents[2] / "test" / "fixtures" / "ltr"
rng = np.random.default_rng(3)
n, q, f = 400, 20, 6
X = rng.normal(size=(n, f))
X[rng.random((n, f)) < 0.2] = np.nan
signal = np.nan_to_num(X[:, 0]) * 1.5 + np.nan_to_num(X[:, 2]) + rng.normal(size=n) * 0.7
y = np.clip(np.floor((signal - signal.min()) / (signal.max() - signal.min()) * 4), 0, 3).astype(int)
names = [f"f{i}" for i in range(f)]
d = xgb.DMatrix(X, y, feature_names=names)
d.set_group([n // q] * q)
for objective in ("rank:pairwise", "rank:ndcg"):
    booster = xgb.train(
        dict(objective=objective, eta=0.3, max_depth=3, min_child_weight=1, seed=0, tree_method="hist"), d, 6
    )
    tag = objective.replace(":", "_")
    booster.save_model(out / f"{tag}.model.json")
    probe = X[:60]
    margin = booster.predict(xgb.DMatrix(probe, feature_names=names), output_margin=True)
    (out / f"{tag}.expected.json").write_text(
        json.dumps(
            {"rows": [[None if np.isnan(v) else float(v) for v in r] for r in probe], "margin": [float(v) for v in margin]}
        )
    )
print("xgboost", xgb.__version__)
