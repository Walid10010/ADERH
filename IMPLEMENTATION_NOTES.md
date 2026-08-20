# Implementation notes — reference semantics of ADERH

The packaged implementation (`aderh/_aderh.py`) is deliberately
**behavior-preserving** with respect to the original `ADERH.py` used for the
NeurIPS 2025 experiments: for identical data and `random_state` it produces
identical scores (see `tests/test_equivalence.py`, atol=1e-12). The published
numbers therefore remain exactly reproducible.

Three aspects of the reference semantics deserve explicit documentation,
because they differ from the most literal reading of the paper text. Before
any re-implementation elsewhere (e.g., a PyOD port), decide consciously
whether to preserve or revise them — revising will change numerical results.

## 1. Pair-partner selection

For each ensemble member, `n` anchor points are sampled without replacement
(`anchor_idx`). The pairing permutation `perm` (a derangement-adjusted
permutation of `0..n-1`) is then applied **to the dataset rows directly**
(`X[perm]`), i.e., partners are always drawn from the first `n` rows of `X`
in their current order — not from the sampled anchors (`anchors[perm]`).
The derangement fix-up (no `perm[j] == j`) only has its intended
"no self-pairing" meaning under the `anchors[perm]` reading.

Consequence: partner points act as fixed landmarks shared across all
ensemble members (up to permutation), and results can depend on dataset row
order. The duplicated sphere centers, however, are `anchors[perm]` — so the
pair *distance* uses `X[perm]` while the second sphere *center* uses
`anchors[perm]`.

## 2. Radius definition in squared space

`sq_pair_dist` holds **squared** Euclidean distances. The reference code
halves this squared quantity (`sq_pair_dist / 2`) to obtain squared radii,
which corresponds to a radius of `d/√2` — not `d/2` as "halving the pairwise
distance" would give (that would be `sq_pair_dist / 4` in squared space).

## 3. Radius–center alignment of the duplicated set

Duplicated centers are stacked block-wise: `[anchors; anchors[perm]]`
(length `2n`). Duplicated squared radii are built **interleaved**:
`[r0, r0, r1, r1, ...]` via `np.repeat(..., 2, axis=1).reshape(-1)`.
Block-wise centers combined with interleaved radii means sphere `j` (for
`j ≥ 1`) is generally assigned the radius of a *different* pair than the one
that produced its center. A block-consistent assignment would be
`np.concatenate([r, r])`.

## 4. Minor reference-code details preserved or cleaned

- Zero squared radii are replaced by `1.0` at scoring time (magic value on
  MinMax-scaled data); preserved.
- `predit()` (typo) kept as a deprecated alias; `outlier_score` kept as an
  alias of `decision_scores_`.
- Removed without behavioral effect: unused functions (`sigmoid`,
  `adjusted_sigmoid`, `abfall`, `_eudis5`, `_normalized_vector`), the
  matplotlib import side effect, unused attributes, and the in-place
  mutation of fitted state during scoring.

## Recommended follow-ups

1. Re-run the ADBench protocol with the two "intended" variants
   (`anchors[perm]` partners; block-consistent radii) and compare AUCs. If
   results are equal or better, publish as v2 semantics with a changelog;
   if worse, the documented reference semantics stand.
2. Only after that decision, open the PyOD pull request — PyOD reviewers
   will read the code closely, and it is better to have this analysis in
   hand than to receive it in review.
