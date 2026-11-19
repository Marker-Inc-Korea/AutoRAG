/**
 * The XGBoost trainer, run as `python <script> <train.tsv> <valid.tsv> <model.json>`.
 *
 * It is embedded because `dist` is all that ships, and it is deliberately small: read two LightGBM-style text files
 * (header, label first, `nan` = missing, group sizes in `<file>.query`), fit a pairwise ranker with early stopping on
 * the validation queries, keep only the trees up to the best round, and print one JSON line with what ran. Judging the
 * model happens in TypeScript on the holdout, with the same predictor that serves it.
 */
export const XGBOOST_TRAIN_SCRIPT = `import json
import sys

import numpy as np
import xgboost as xgb


def load(path):
    with open(path, encoding="utf-8") as handle:
        header = handle.readline().rstrip("\\n").split("\\t")
        rows = [line.rstrip("\\n").split("\\t") for line in handle if line.strip()]
    data = np.array([[float(value) for value in row] for row in rows], dtype=np.float64)
    with open(path + ".query", encoding="utf-8") as handle:
        groups = [int(line) for line in handle if line.strip()]
    matrix = xgb.DMatrix(data[:, 1:], data[:, 0], feature_names=header[1:])
    matrix.set_group(groups)
    return matrix


train = load(sys.argv[1])
valid = load(sys.argv[2])
params = dict(
    objective="rank:pairwise",
    eval_metric="ndcg@10",
    eta=0.05,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    seed=0,
    nthread=4,
    tree_method="hist",
)
booster = xgb.train(params, train, 600, evals=[(valid, "valid")], early_stopping_rounds=50, verbose_eval=False)
best = booster.best_iteration
booster[: best + 1].save_model(sys.argv[3])
print(json.dumps({"xgboost": xgb.__version__, "bestIteration": int(best)}))
`;
