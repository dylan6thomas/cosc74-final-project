# Do Fraud Detectors Learn Fraud or Shortcuts?

COSC 74 final project (Dylan Thomas). This project looks at how much fake-job-ad detectors trained on EMSCAD rely on easy-to-fake metadata (a company logo, a company profile) rather than the content of the ad, and how well they hold up under distribution shift.

## Contents
- `fraud_detection.ipynb`: full pipeline (preprocessing, grouped splits, models B0/B1/M1–M4, CV tuning, test evaluation, ablation, SHAP, counterfactual test, region/industry shift, threshold and error analysis)
- `data/fake_job_postings.csv`: EMSCAD (Vidros et al., 2017), Kaggle redistribution by S. Bansal, CC0
- `figures/`, `results/`: outputs written by the notebook
- `requirements.txt`: pinned dependencies

## Reproduce
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name cosc74-fraud --display-name "COSC74 Fraud (.venv)"
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 fraud_detection.ipynb
```
Run from this folder (the notebook uses relative paths). All randomness is seeded (`SEED = 42`). The first run downloads `all-MiniLM-L6-v2` and caches the embeddings in `cache/`. Fitted models are saved to `models/`; that folder is not tracked.
