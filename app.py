import json
from pathlib import Path

import streamlit as st

from src.paths import ARTIFACTS_DIR, DATA_DIR, MODELS_DIR, PROJECT_ROOT

HORIZONS = [24, 72, 168]


def model_path_for_horizon(h: int) -> Path:
    path = MODELS_DIR / f"bilstm_{h}h.pth"
    if path.exists():
        return path
    if h == 72:
        fallback = MODELS_DIR / "bilstm_model.pth"
        if fallback.exists():
            return fallback
    return path


def load_metrics(horizon: int) -> dict | None:
    path = ARTIFACTS_DIR / f"metrics_{horizon}h.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def show_png(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Not found: `{path.name}`")


@st.cache_data(show_spinner=False)
def run_evaluation(horizon: int, model_path: str, model_mtime: float) -> dict:
    del model_mtime
    from src.evaluate import evaluate

    return evaluate(output_window=horizon, model_path=model_path)


def render_metrics(metrics: dict | None) -> None:
    if metrics is None:
        st.caption("No saved metrics for this horizon. Run evaluation from the sidebar.")
        return
    cols = st.columns(5)
    cols[0].metric("RMSE", f"{metrics['rmse']:.2f} W/m²")
    cols[1].metric("nRMSE", f"{metrics['nrmse']:.2%}")
    cols[2].metric("MAE", f"{metrics['mae']:.2f} W/m²")
    cols[3].metric("R²", f"{metrics['r2']:.4f}")
    cols[4].metric("Skill score", f"{metrics['skill_score']:.2f}")


def main() -> None:
    st.set_page_config(page_title="Solar GHI LSTM", layout="wide")
    st.title("Solar irradiance LSTM — results")

    with st.sidebar:
        st.header("Settings")
        horizon = st.selectbox("Forecast horizon (hours)", HORIZONS, index=1)
        model_path = model_path_for_horizon(horizon)
        st.text(f"Model: {model_path.relative_to(PROJECT_ROOT)}")
        st.text(f"Data: {DATA_DIR}")
        run_eval = st.button("Run evaluation", type="primary")

    if run_eval:
        if not model_path.exists():
            st.error(
                f"Model not found at `{model_path}`. "
                f"Train with: `PYTHONPATH=. .venv/bin/python src/train.py "
                f"--output_window {horizon} --model_path {model_path}`"
            )
        else:
            with st.spinner(f"Evaluating {horizon}h horizon…"):
                try:
                    mtime = model_path.stat().st_mtime
                    metrics = run_evaluation(
                        horizon, str(model_path), mtime
                    )
                    st.session_state["metrics"] = metrics
                    st.success("Evaluation complete.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")

    metrics = st.session_state.get("metrics") or load_metrics(horizon)
    if metrics and metrics.get("output_window") != horizon:
        metrics = load_metrics(horizon)

    st.subheader(f"Metrics ({horizon}h)")
    render_metrics(metrics)

    tab_eda, tab_train, tab_ghi, tab_pv = st.tabs(
        ["EDA", "Training", "GHI forecast", "PV"]
    )

    with tab_eda:
        show_png(ARTIFACTS_DIR / "eda_ghi_distribution.png", "GHI distribution by site")
        show_png(ARTIFACTS_DIR / "eda_diurnal_cycle.png", "Average diurnal GHI profile")
        show_png(ARTIFACTS_DIR / "eda_correlation.png", "Feature correlation heatmap")

    with tab_train:
        show_png(ARTIFACTS_DIR / f"loss_{horizon}h.png", f"Training loss ({horizon}h)")
        show_png(ARTIFACTS_DIR / "loss_curve.png", "Training loss (legacy)")

    with tab_ghi:
        show_png(
            ARTIFACTS_DIR / f"pred_vs_actual_{horizon}h.png",
            f"Actual vs predicted GHI ({horizon}h)",
        )
        show_png(
            ARTIFACTS_DIR / f"scatter_{horizon}h.png",
            f"Predicted vs actual scatter ({horizon}h)",
        )
        show_png(ARTIFACTS_DIR / "forecast_samples.png", "Forecast samples (legacy)")

    with tab_pv:
        show_png(ARTIFACTS_DIR / f"pv_power_forecast_{horizon}h.png", f"PV forecast ({horizon}h)")
        show_png(ARTIFACTS_DIR / "pv_power_forecast.png", "PV forecast (legacy)")
        show_png(ARTIFACTS_DIR / "plant_correlation.png", "Plant weather correlation")


if __name__ == "__main__":
    main()
