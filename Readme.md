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


##  Run Experiments



```bash
python run_experiment.py --data_root ./data --out_dir ./outputs
```

