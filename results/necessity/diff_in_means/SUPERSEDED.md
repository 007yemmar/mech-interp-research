# SUPERSEDED by `results/necessity/direction_audit/`

These artifacts come from the **unwhitened** difference-in-means baseline —
`d_c = mean(X[y_c=1]) - mean(X[y_c=0])`, measured in raw activation units.
Do not cite them as "the difference-in-means baseline". They are retained only
as the control arm showing what the plain estimator does.

**Why they are not a usable baseline.** The 46 per-code directions are not 46
directions: their mean pairwise |cos| is 0.685 and the effective dimensionality
of the set is **1.89**. They collapse onto ~2 shared axes, which is why the
off-target profile was flat (`mean_abs_off_r` 0.093 ± 0.011 across all 46 codes)
and the specificity ratio was 1.25 — an "on-target" correlation barely
distinguishable from an off-target one. On-target median |r| was 0.121, below a
logistic-regression probe on the identical pooled features (LR-equivalent median
|r| = 0.307, beating this baseline on 46/46 codes).

**What replaced them.** `results/necessity/direction_audit/`, built by
`modal_app/diff_in_means_directions.py` and audited by
`modal_app/necessity_audit.py` on the shared harness. Three arms, `d_eff = M^-1 d`:

| arm | M | median on-target \|r\| | effective dims of the 46 directions |
|---|---|---|---|
| `diff_in_means_none` | I | 0.121 | 1.89 |
| `diff_in_means_diagonal` | diag(Σ) | 0.127 | 1.82 |
| `diff_in_means_full` | Σ (Ledoit-Wolf) | **0.339** | **33.68** |

The `full` arm is the mass-mean probe of Marks & Tegmark (2023). It beats the
LR ceiling on 39/46 codes and is what the paper should cite.

Note also that these artifacts were produced by `diff_in_means_baseline.run_diff_in_means_baseline`,
which derives its own code panel by prevalence and computes its own off-target
statistics. The replacement routes through `necessity_audit` with the pinned
46-code panel, so it is on the same code path as every other source.
