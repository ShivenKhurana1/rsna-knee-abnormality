import numpy as np
from sklearn.metrics import roc_auc_score

HERE = "/home/zha76451/Projects/rsna-knee-abnormality/oof_harness"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

F_dino = np.load(f"{HERE}/F_dino.npy")
F_rad11 = np.load(f"{HERE}/F_rad11.npy")
F_rad13 = np.load(f"{HERE}/F_rad13.npy")
F_v52 = np.load(f"{HERE}/F_v52.npy")
gold_mask = np.load(f"{HERE}/gold_mask.npy")
Y = np.load(f"{HERE}/Y_all.npy")
weak_mask = ~gold_mask
rng = np.random.default_rng(20260829)


def nan_safe_blend(weights, families, rows):
    """weights: dict name->w, families: dict name->(N,12) array. Per-row, per-target
    average over whichever families are non-NaN, weights renormalized among those present."""
    stack = np.stack([families[n][rows] for n in weights], axis=0)   # (F, n, 12)
    w = np.array([weights[n] for n in weights], dtype=np.float64)[:, None, None]
    valid = ~np.isnan(stack)
    wv = w * valid
    denom = wv.sum(axis=0)
    num = np.nan_to_num(stack, nan=0.0) * wv
    out = np.divide(num.sum(axis=0), denom, out=np.full(denom.shape, np.nan), where=denom > 0)
    return out


def macro_auc(pred, Yrows, per_target=False):
    aucs = {}
    for j, name in enumerate(LAB):
        valid = ~np.isnan(pred[:, j])
        yv, pv = Yrows[valid, j], pred[valid, j]
        if len(set(yv.astype(int))) > 1:
            aucs[name] = roc_auc_score(yv, pv)
    m = float(np.mean(list(aucs.values())))
    return (m, aucs) if per_target else m


FAMS = {"dino": F_dino, "e11": F_rad11, "e13": F_rad13, "v52": F_v52}

print("=== Reproduce documented family blends, n=4349 (weak pool) ===")
combos = {
    "e13 alone": {"e13": 1.0},
    "e11+e13 equal": {"e11": 0.5, "e13": 0.5},
    "e11+e13+v52 equal": {"e11": 1 / 3, "e13": 1 / 3, "v52": 1 / 3},
    "dino+e11+e13+v52 equal": {"dino": 0.25, "e11": 0.25, "e13": 0.25, "v52": 0.25},
}
weak_rows = np.where(weak_mask)[0]
Yw = Y[weak_rows]
results = {}
for name, w in combos.items():
    pred = nan_safe_blend(w, FAMS, weak_rows)
    au = macro_auc(pred, Yw)
    results[name] = au
    print(f"  {name:26s} {au:.4f}")

print("\n=== The open question: does adding dino to the deployed e11+e13+v52 blend help? ===")
print("(dino only covers 60% of studies -- comparison restricted to that subset for a fair test)")
dino_rows = weak_rows[~np.isnan(F_dino[weak_rows, 0])]
Yd = Y[dino_rows]
base_pred = nan_safe_blend({"e11": 1 / 3, "e13": 1 / 3, "v52": 1 / 3}, FAMS, dino_rows)
base_au, base_per = macro_auc(base_pred, Yd, per_target=True)
print(f"  e11+e13+v52 equal, on dino-covered subset (n={len(dino_rows)}): {base_au:.4f}")

for dw in (0.10, 0.15, 0.20, 0.25):
    rest = (1 - dw) / 3
    w = {"dino": dw, "e11": rest, "e13": rest, "v52": rest}
    pred = nan_safe_blend(w, FAMS, dino_rows)
    au = macro_auc(pred, Yd)
    print(f"  +dino at weight {dw:.2f}: {au:.4f}  (delta {au - base_au:+.4f})")

# Paired bootstrap on the best-looking dino weight vs the no-dino baseline, same subset
best_dw = max((0.10, 0.15, 0.20, 0.25),
              key=lambda dw: macro_auc(nan_safe_blend(
                  {"dino": dw, "e11": (1 - dw) / 3, "e13": (1 - dw) / 3, "v52": (1 - dw) / 3},
                  FAMS, dino_rows), Yd))
w_best = {"dino": best_dw, "e11": (1 - best_dw) / 3, "e13": (1 - best_dw) / 3, "v52": (1 - best_dw) / 3}
cand_pred = nan_safe_blend(w_best, FAMS, dino_rows)
cand_au, cand_per = macro_auc(cand_pred, Yd, per_target=True)

print(f"\n=== Paired bootstrap: e11+e13+v52+dino(w={best_dw}) vs e11+e13+v52, n={len(dino_rows)} ===")
n = len(dino_rows)
deltas = []
B = 2000
for _ in range(B):
    idx = rng.integers(0, n, n)
    b_base = macro_auc(base_pred[idx], Yd[idx])
    b_cand = macro_auc(cand_pred[idx], Yd[idx])
    deltas.append(b_cand - b_base)
deltas = np.array(deltas)
lo, hi = np.percentile(deltas, [2.5, 97.5])
p_better = float((deltas > 0).mean())
print(f"  observed delta: {cand_au - base_au:+.4f}")
print(f"  95% CI: [{lo:+.4f}, {hi:+.4f}]")
print(f"  P(better): {p_better:.1%}")

print("\n  per-target deltas (candidate - baseline):")
for name in LAB:
    d = cand_per.get(name, float("nan")) - base_per.get(name, float("nan"))
    print(f"    {name:18s} {d:+.4f}")

print("\n=== Sanity: same comparison on the 58-study gold gate (not resolving power, reference only) ===")
gold_rows = np.where(gold_mask)[0]
gold_dino_rows = gold_rows[~np.isnan(F_dino[gold_rows, 0])]
Yg = Y[gold_dino_rows]
base_g = macro_auc(nan_safe_blend({"e11": 1 / 3, "e13": 1 / 3, "v52": 1 / 3}, FAMS, gold_dino_rows), Yg)
cand_g = macro_auc(nan_safe_blend(w_best, FAMS, gold_dino_rows), Yg)
print(f"  n={len(gold_dino_rows)}  baseline {base_g:.4f}  candidate {cand_g:.4f}  delta {cand_g - base_g:+.4f}")
