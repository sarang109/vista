# Optional real-data backbone: PaySim

The default workload is generated synthetically (see `src/vista/simulator/workload.py`).
To drive the pipeline with the public PaySim mobile-money transaction log instead:

1. Download `PS_20174392719_1491204439457_log.csv` from Kaggle (dataset "Synthetic Financial Datasets For
   Fraud Detection", E. Lopez-Rojas). A free Kaggle account is required; no API key is needed by VISTA.
2. Save it as `data/paysim.csv` (this path is set by `world.paysim_path` in the configs).
3. Re-run `python -m vista.run_all --config configs/experiments/full.json`.

What is taken from PaySim: amounts, timing (`step` = hour), counterparties (`nameOrig`, `nameDest`) and
`isFraud` (PaySim models account-takeover-then-cash-out, mapped to the `ato_like` scenario). What is
still synthetic: bank membership, security evidence (authentication, device, credential lifecycle), because
no public dataset contains interbank authentication evidence. PaySim is itself a simulator calibrated on
real logs, so it is "transaction-like", not a real bank's data.


## RBA login dataset (E26)

`python -m vista.run_all --only e26 --config configs/experiments/full.json` downloads
`https://zenodo.org/records/6782156/files/rba-dataset.zip` to `data/rba/` (1.1 GB, MD5
`cc1b1078b3929650e6c08678caffcc57`, CC BY 4.0) and builds `data/rba_cache/` (per-login chain states for a 6 % stable
user sample). Neither directory is committed. Cite: S. Wiefling, P. R. Jørgensen, S. Thunem, L. Lo Iacono, *Pump Up
Password Security! Evaluating and Enhancing Risk-Based Authentication on a Real-World Large-Scale Online Service*,
ACM TOPS 2022, doi:10.1145/3546069.
