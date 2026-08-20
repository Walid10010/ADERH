# ADERH — Anomaly Detection by an Ensemble of Random Pairs of Hyperspheres

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://neurips.cc/virtual/2025/poster/115418)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![tests](https://github.com/Walid10010/ADERH/actions/workflows/ci.yml/badge.svg)](https://github.com/Walid10010/ADERH/actions)

Official implementation of the NeurIPS 2025 paper
**"Anomaly Detection by an Ensemble of Random Pairs of Hyperspheres"**
(Walid Durani, Collin Leiber, Khalid Durani, Claudia Plant, Christian Böhm).

ADERH is a **fast**, **hyperparameter-robust**, isolation-based unsupervised
anomaly detector for tabular data. Guided by a δ-separation argument, it
covers normal regions with an ensemble of small hyperspheres built from
randomly paired points; each sphere's isolation signal is refined by
**Pitch** (a boundary-proximity ratio) and **NDensity** (a sparsity-aware
density weight), and signals are averaged over the ensemble.

## Install

```bash
pip install aderh
```

or from source:

```bash
git clone https://github.com/Walid10010/ADERH.git && cd ADERH
pip install -e .
```

Runtime dependencies: `numpy`, `scikit-learn` only.

## Quickstart

```python
from aderh import ADERH

det = ADERH(random_state=0).fit(X_train)      # unsupervised
scores = det.decision_function(X_test)        # higher = more anomalous
labels = det.predict(X_test)                  # 1 = anomaly, 0 = normal
```

Key parameters: `n_estimators=256` (ensemble size), `n=18` (hyperspheres per
member), `contamination=0.1` (sets the label threshold). Defaults reproduce
the paper.

## Reproducing the paper

Experiments are based on the [ADBench benchmark](https://github.com/Minqi824/ADBench):

```bash
pip install -r requirements-experiments.txt
git clone https://github.com/Minqi824/ADBench.git && mv ADBench data
python experiments/run_experiment.py
```

Protocol: MinMax scaling to [0, 1], 3 stratified 70/30 splits
(`StratifiedShuffleSplit`, `random_state=0`), seeds `0, 1, 2, 1000, 10000`
for stochastic methods, AUC-ROC and AUC-PR averaged over splits × seeds,
results appended to `results.csv`.

## Implementation notes

The packaged detector is verified **score-identical** to the original
reference script (`tests/test_equivalence.py`). Scoring convention follows
PyOD: higher `decision_function` values indicate anomalies
(`score_samples` provides the scikit-learn sign convention). Hypersphere
radii are defined in squared-distance space; see
[IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) for the precise reference
semantics.

## Citation

```bibtex
@inproceedings{durani2025aderh,
  title     = {Anomaly Detection by an Ensemble of Random Pairs of Hyperspheres},
  author    = {Durani, Walid and Leiber, Collin and Durani, Khalid and
               Plant, Claudia and B{\"o}hm, Christian},
  booktitle = {Advances in Neural Information Processing Systems 38 (NeurIPS)},
  year      = {2025}
}
```

## License

Released under the [MIT License](LICENSE).
