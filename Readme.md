# ADERH — Anomaly Detection by an Ensemble of Random Pairs of Hyperspheres

**Paperpage:** https://neurips.cc/virtual/2025/poster/115418


##  Environment Setup

- **Python:** 3.7  
- **Dependencies:** Install via requirements file.

```bash
pip install -r requirements.txt
````


##  Dataset

Experiments are based on the **[ADBench benchmark](https://github.com/Minqi824/ADBench/tree/main)**.

Download and prepare data:

```bash
git clone https://github.com/Minqi824/ADBench.git
mv ADBench ./data
```

Expected structure:

```
ADERH/
│
├── ADERH.py
├── run_experiment.py
├── requirements.txt
└── Classical/
    ├── dataset1.npz
    ├── dataset2.npz
    └── ...
```

---

##  Run Experiments

Reproduce the NeurIPS paper results:

```bash
python run_experiment.py
```

### Script details

* Automatically scans datasets under `Classical/*`.
* Normalizes features with **MinMaxScaler** to [0, 1].
* Performs **3 stratified 70/30 train-test splits** (`StratifiedShuffleSplit`).
* Uses **random seeds = [0, 1, 2, 100, 1000]** for stochastic algorithms.
* Computes **AUC-ROC** and **Average Precision (AUC-PR)**.
* Results are printed to console and appended to **`kdd2025_all_repeat.csv`**.




##  Evaluation Protocol

* **Normalization:** MinMax [0, 1]
* **Splits:** 3 stratified (70/30)
* **Seeds:** 0, 1, 2, 100, 1000 (for stochastic methods)
* **Metrics:** AUC-ROC and AUC-PR (averaged over splits and seeds)
* **Output:** results appended to `results.csv`



