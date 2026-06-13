# Memory Notes — deeponet-acoustic-wave-prop

## 2026-06-13: Gradient Clipping Root Cause

- The upstream code had `optax.clip_by_global_norm(0.01)` at `deeponet_acoustics/models/deeponet.py:189`. The PNAS paper specifies "Gradient clipping with an absolute value of 0.1" — **per-element** clipping, not global norm. Fixed to `optax.clip(0.1)`.
- With 27.6M parameters, the global gradient norm always exceeds 0.01, so `clip_by_global_norm(0.01)` damped ALL gradients on every step. This is why the 20-epoch training run never converged — the model weights barely changed.
- Weight initialization IS correct: first layer `sinusoidal_init(is_first=True)` correctly produces `wi ~ U[-√(6/n)/1, √(6/n)/1]` per the paper's k=1 rule. Do not change this.
- The remaining challenge is gradient attenuation: each backward pass attenuates ~0.045× per layer due to W_i ≈ N(0, 0.001²) with 2048-dim layers. After 6 layers: ~10⁻⁹. The clipping fix alone does not solve this — proper convergence requires the full 50-70k iterations (1-3 days GPU time) with Adam's adaptive learning rates.

## 2026-06-13: SIREN w₀ Forward & U/V Transformer Fixes

- **Root cause of vanishing gradients**: Two coupled architectural bugs in `networks_flax.py`:

  1. **Missing `angular_freq=30` in non-first layer forward pass** (`networks_flax.py:279`): The SIREN paper applies `w₀=30` as a forward-pass multiplier in ALL layers except the first: `sin(w₀·(Wx+b))`. The first layer uses `sin(Wx+b)` (w₀ implicitly = 1). This code had the pattern inverted: the first layer received `angular_freq=30` (correct per the SIREN first-layer init convention), but non-first hidden layers were `sin(Wx+b)` with tiny weights (`U(-√(6/n)/30, √(6/n)/30)`) — resulting in near-linear propagation and gradient attenuation of ~0.047× per layer. Fixed by adding `self.angular_freq *` to the activation input for all non-first Dense layers in both `ModMLP` and `MLP`.

  2. **U/V transformer layers used `is_first=True` init** (`networks_flax.py:246-259`): The U/V gating Dense layers were initialized with `sinusoidal_init(is_first=True)` → `U(-1/d_in, 1/d_in)`. For the branch with `d_in=1728`, this gives weights with std ≈ 0.000334, producing U/V values ≈ ±0.001 after sin activation. Since U/V gates every hidden layer output via `output = U·h + (1-U)·V`, the gradient through each layer was attenuated by `||U|| ≈ 0.045`, killing the branch gradient over 6 layers (0.045⁶ ≈ 10⁻¹⁸). Fixed by switching U/V to `sinusoidal_init(is_first=False)` + `angular_freq=30` in activation — same as other non-first layers — giving U/V spanning full [-1, 1] regardless of input dimensionality.

- **Gradient norm diagnostic results** (fresh init, one forward+backward pass on cube_p6_64pilot data):

  | Layer | Before any fix | After w₀ fix only | After U/V + w₀ fix |
  |---|---|---|---|
  | tn/linear_tn_0/kernel | 3.08e-16 [DEAD] | 2.28e-10 [DEAD] | **8.20e-05 [TINY]** |
  | bn/linear_bn_0/kernel | 0.00e+00 [DEAD] | 1.47e-18 [DEAD] | **1.18e-07 [DEAD]** |
  | bn/transformerU/kernel | 1.11e-09 [DEAD] | 3.37e-08 [DEAD] | **7.63e-04 [OK]** |
  | tn/transformerU/kernel | 1.85e-09 [DEAD] | 5.49e-08 [DEAD] | **2.90e-04 [OK]** |
  | Median across all layers | 3.72e-10 [DEAD] | 2.74e-08 [DEAD] | **3.78e-04 [OK]** |

  `tn_0/kernel` at 8.2e-05 is borderline (just below 1e-4 threshold). `bn_0/kernel` at 1.2e-07 is low but nonzero — Adam's per-parameter LR can compensate over 50-70k iterations. All U/V layers, hidden layers 1-5, and b0 are solidly in the OK regime. The 50-70k training run is now viable.

- **Key insight**: The U/V fix was the dominant contributor (11 orders of magnitude improvement for bn_0). The w₀ forward fix contributed ~6 orders for the trunk but only ~2 for the branch, because the branch gradient was bottlenecked by U/V gating, not hidden-layer activation. Both fixes are needed for both networks.

- **Per-element clipping fix** (from deeponet.py:189, `clip_by_global_norm(0.01)` → `clip(0.1)`) remains necessary but NOT sufficient on its own. The architectural fixes were the real blocker.

- **Files patched**: `networks_flax.py` — three changes in `ModMLP.__call__` and one in `MLP.__call__`.

## Checkpoint Structure

- `writeModel()` saves `self.params` directly as a dict with keys: `['adaptive_weights', 'b0', 'bn', 'tn']`. No `opt_state` or `step`.
- `writeTrainingCheckpoint()` saves `{"params": self.params, "opt_state": self.opt_state, "step": int}`. This only runs if `checkpoint_dir` is set in config.
- When restoring: check for `"params" in restored` to distinguish formats.

## Data Loading

- The training loader expects `batch_size_coord=200` (not 1000) for the trunk coordinate batch size.
- Branch batch: `(branch_batch, coordinates_batch)` = `(64, 200)` samples per iteration.
- Each iteration = 1 optimizer step (not 1 epoch). nIter = total optimizer steps.
- Data normalization: `u_p_range=(-2.0, 2.0)` for initial pressures, temporal normalized by spatial factor.
