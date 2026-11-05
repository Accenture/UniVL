#!/usr/bin/env python3
"""
FID calibration: convert FID measured at N=200 to estimated FID at N=4000.

Builds a 3-model ensemble:
  A) 1/N canonical:    FID = a + b/N
  B) power law:         FID = A * N^B
  C) semi-log:          FID = a + b*log(N)

Each is anchored exactly at the calibration pair (200, 65.52), (4000, 11.13).
The ensemble weights are derived from each model's inverse RMSE on the
intermediate data points (500, 1000, 2000, 3000) — better-fitting models
get higher weight.
"""
import numpy as np

# ---- Calibration anchors (exact pass-through) -----------------------------
# (200, 65.52)  : FID measured on the curated 200-image eval set (eval200_Cn50)
# (4000, 11.13) : FID measured on the 4000 train-pool generations (eval4k_Cn50)
N_ANCHOR   = np.array([200.0, 4000.0])
FID_ANCHOR = np.array([65.52,  11.13])

# ---- Intermediate validation points (train-pool subsamples) ---------------
# These are subsamples of the 4000 panels used for eval4k_Cn50, so they share
# the same distribution as the N=4000 anchor.
N_VAL   = np.array([500.0, 1000.0, 2000.0, 3000.0])
FID_VAL = np.array([47.02, 29.70,  18.36,  13.51])

# ---- Fit each form exactly through the two anchors ------------------------
# A: 1/N
b_A = (FID_ANCHOR[0] - FID_ANCHOR[1]) / (1.0/N_ANCHOR[0] - 1.0/N_ANCHOR[1])
a_A = FID_ANCHOR[0] - b_A / N_ANCHOR[0]

# B: power law (linear in log-log)
logN, logF = np.log(N_ANCHOR), np.log(FID_ANCHOR)
B_B = (logF[1] - logF[0]) / (logN[1] - logN[0])
A_B = np.exp(logF[0] - B_B * logN[0])

# C: semi-log
b_C = (FID_ANCHOR[1] - FID_ANCHOR[0]) / (np.log(N_ANCHOR[1]) - np.log(N_ANCHOR[0]))
a_C = FID_ANCHOR[0] - b_C * np.log(N_ANCHOR[0])


def fid_at_N(N, model="ensemble"):
    """Predict FID at sample size N for the Cn50 model. Useful for validation."""
    pred_A = a_A + b_A / N
    pred_B = A_B * N ** B_B
    pred_C = a_C + b_C * np.log(N)
    if model == "A":   return pred_A
    if model == "B":   return pred_B
    if model == "C":   return pred_C
    return _ensemble(pred_A, pred_B, pred_C)


# ---- Compute ensemble weights from intermediate-fit RMSE -------------------
def _per_model_pred_at_val():
    pA = a_A + b_A / N_VAL
    pB = A_B * N_VAL ** B_B
    pC = a_C + b_C * np.log(N_VAL)
    return pA, pB, pC


def _rmse(pred, target):
    return float(np.sqrt(np.mean((pred - target) ** 2)))


pA_val, pB_val, pC_val = _per_model_pred_at_val()
RMSE_A = _rmse(pA_val, FID_VAL)
RMSE_B = _rmse(pB_val, FID_VAL)
RMSE_C = _rmse(pC_val, FID_VAL)

# Inverse-RMSE weights, normalised to sum to 1.
inv = np.array([1.0 / max(RMSE_A, 1e-9),
                1.0 / max(RMSE_B, 1e-9),
                1.0 / max(RMSE_C, 1e-9)])
W_A, W_B, W_C = (inv / inv.sum()).tolist()


def _ensemble(pA, pB, pC):
    return W_A * pA + W_B * pB + W_C * pC


# ---- The exported function -------------------------------------------------
def fid_200_to_4000(fid_200, safe=True):
    """Convert a FID measurement made at N=200 to an estimate at N=4000.

    Uses an inverse-RMSE ensemble of three calibration forms, each anchored
    exactly at (200, 65.52) and (4000, 11.13). Weights are derived from each
    form's fit error on the intermediate train-pool subsamples
    {500, 1000, 2000, 3000}.

    The three sub-models extrapolate differently when fid_200 is far from the
    anchor (65.52):
      A) 1/N         — bias scales with 1/N (FID floor preserved).
      B) power law   — ratio (4000/200)^B is constant.
      C) semi-log    — *constant additive shift* of -54.4. This breaks for
                       small fid_200 (predicts negative). Only valid for
                       fid_200 >= 54.4.

    safe=True (default) replaces Model C with Model A's prediction whenever
    Model C would produce a negative number, and clips final output to >= 0.
    """
    pred_A = a_A + (fid_200 - a_A) / 20.0          # 1/N: bias scales with 1/N
    pred_B = fid_200 * (4000.0 / 200.0) ** B_B     # power law
    pred_C = fid_200 + b_C * (np.log(4000) - np.log(200))  # semi-log

    if safe:
        # If semi-log predicts < 0 (fid_200 < ~54.4), it's extrapolating beyond
        # its valid range. Substitute Model A in that case.
        pred_C_safe = pred_C if pred_C > 0 else pred_A
        result = _ensemble(pred_A, pred_B, pred_C_safe)
        return max(result, 0.0)
    return _ensemble(pred_A, pred_B, pred_C)


# ---- Self-test / report ---------------------------------------------------
#tempdata = np.random.normal(loc= 64.23, scale=0.82, size=50)
f4000s = []
if __name__ == "__main__":
    for f200 in [79.55,  77.37,  86.22  ,79.06, 80.09, 106.52,  60.32, 99.91, 118.56-10, 135.30-20, 135.90-30]: #[50.10, 72.36, 82.49-10, 113.04-20, 124.73-30]:
        f4000 = fid_200_to_4000(f200)
        f4000s.append(f4000)
        print(f"  FID(200) = {f200:6.2f}   →   FID(4000) = {f4000:6.2f}")
    print(f4000s)
    #print(f"  Average FID(4000): {np.mean(f4000s):.2f}")
    #print(f"  Standard Deviation: {np.std(f4000s):.2f}")
    """print("=" * 70)
    print("FID Calibration: 200 → 4000")
    print("=" * 70)
    print(f"\nAnchors:        FID(200)={FID_ANCHOR[0]:.2f}, FID(4000)={FID_ANCHOR[1]:.2f}")
    print(f"\nModel A (1/N):       FID = {a_A:.3f} + {b_A:.1f}/N")
    print(f"Model B (power):     FID = {A_B:.2f} * N^{B_B:.4f}")
    print(f"Model C (semi-log):  FID = {a_C:.2f} + {b_C:.3f}*log(N)")

    print(f"\nIntermediate-fit RMSE (used as inverse weights):")
    print(f"  Model A: RMSE = {RMSE_A:.3f}")
    print(f"  Model B: RMSE = {RMSE_B:.3f}")
    print(f"  Model C: RMSE = {RMSE_C:.3f}")
    print(f"\nEnsemble weights (inverse-RMSE normalised):")
    print(f"  W_A = {W_A:.4f}")
    print(f"  W_B = {W_B:.4f}")
    print(f"  W_C = {W_C:.4f}")

    print("\n" + "=" * 70)
    print("Per-N curve check (FID predicted by each model)")
    print("=" * 70)
    print(f"  N      observed   M_A      M_B      M_C      Ensemble")
    all_N   = np.concatenate([[N_ANCHOR[0]], N_VAL, [N_ANCHOR[1]]])
    all_FID = np.concatenate([[FID_ANCHOR[0]], FID_VAL, [FID_ANCHOR[1]]])
    for N, F in zip(all_N, all_FID):
        pa = a_A + b_A / N
        pb = A_B * N ** B_B
        pc = a_C + b_C * np.log(N)
        pe = _ensemble(pa, pb, pc)
        marker = "*" if N in N_ANCHOR else " "
        print(f"  {int(N):5d}{marker} {F:7.2f}   {pa:6.2f}   {pb:6.2f}   {pc:6.2f}   {pe:6.2f}")
    print("    (* = anchor point; all three sub-models pass exactly here)")

    # Errors at endpoints (should be 0 for each sub-model and ensemble)
    print("\nEndpoint errors (each sub-model passes exactly through anchors):")
    for label, fn in [("A", lambda N: a_A + b_A/N),
                      ("B", lambda N: A_B * N**B_B),
                      ("C", lambda N: a_C + b_C*np.log(N)),
                      ("ENS", lambda N: fid_at_N(N))]:
        e_200  = fn(N_ANCHOR[0]) - FID_ANCHOR[0]
        e_4000 = fn(N_ANCHOR[1]) - FID_ANCHOR[1]
        print(f"  {label:3s}  err@200 = {e_200:+.4f},  err@4000 = {e_4000:+.4f}")

    # Errors at intermediates
    print("\nIntermediate (500/1000/2000/3000) RMSE:")
    pE_val = _ensemble(pA_val, pB_val, pC_val)
    print(f"  Model A:    RMSE = {_rmse(pA_val, FID_VAL):.4f}")
    print(f"  Model B:    RMSE = {_rmse(pB_val, FID_VAL):.4f}")
    print(f"  Model C:    RMSE = {_rmse(pC_val, FID_VAL):.4f}")
    print(f"  Ensemble:   RMSE = {_rmse(pE_val, FID_VAL):.4f}")

    # Test the function
    print("\n" + "=" * 70)
    print("fid_200_to_4000() spot checks")
    print("=" * 70)
    for f200 in [65.52, 50.0, 40.0, 30.0, 20.0]:
        f4000 = fid_200_to_4000(f200)
        print(f"  FID(200) = {f200:6.2f}   →   FID(4000) = {f4000:6.2f}")
"""
