# Solar irradiance LSTM

BiLSTM forecasting of global horizontal irradiance (GHI) from BSRN radiation station data, with optional PV power experiments.

## Setup

```bash
cd solar-irradiance-lstm
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Radiation `.tab` files should live in the project root (or set `SOLAR_DATA_DIR` to another directory containing `*_radiation_*.tab` files).

## Running the dashboard

From the project root with the venv activated:

```bash
export PYTHONPATH=.
streamlit run app.py
```

The app shows saved plots under `artifacts/` and lets you re-run model evaluation from the sidebar (updates metrics JSON and forecast plots).

## Regenerating artifacts

```bash
export PYTHONPATH=.

# EDA plots
python src/eda.py

# Train and evaluate (example: 72h horizon)
python src/train.py --output_window 72 --model_path models/bilstm_72h.pth
python src/evaluate.py --output_window 72 --model_path models/bilstm_72h.pth

# Or run all horizons
python src/run_experiments.py
```

Metrics are written to `artifacts/metrics_{horizon}h.json` after evaluation.

Optional environment variables:

- `SOLAR_DEVICE=cpu` — force CPU inference (avoids MPS issues on some Mac builds).
- `OMP_NUM_THREADS=1` — recommended when running evaluation after LightGBM training (avoids OpenMP deadlocks with PyTorch).

Evaluation loads radiation data before importing PyTorch for stability.
