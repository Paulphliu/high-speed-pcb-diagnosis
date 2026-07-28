"""
run_lot_diagnosis.py

Direct-run PN06 lot-aware analysis script.

V5 background-aware primary/secondary signal-aware revision:
- Keep the V3 measurement-level models unchanged.
- Do NOT force every single point into a precise root cause.
- Convert Stage-2 exact predictions plus Stage-1/raw features into coarse
  point-level signal types.
- Let Stage 3 use the lot/board/region/test-point distribution to identify
  systematic issues that are weak or ambiguous at a single measurement point.
- Output primary_lot_diagnosis, secondary_lot_signal, and final_lot_diagnosis
  so mixed lots can be reported as a dominant primary issue with secondary
  evidence instead of being treated only as a single flat label.

Purpose
-------
Use the V3 two-stage measurement-level models to analyze the V4 PN06
lot-aware dataset:

1. Load V3 trained model bundle.
2. Load V4 PN06 lot-aware features.
3. Predict each measurement point.
4. Convert exact point prediction into signal-aware coarse point labels.
5. Aggregate by lot / board / region / repeated test point.
6. Produce Stage 3 PN06 lot diagnosis.

This is a direct-run script for VS Code. Change USER SETTINGS if needed.
"""

from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
except Exception:  # pragma: no cover
    accuracy_score = None
    confusion_matrix = None
    classification_report = None


# ============================================================
# USER SETTINGS
# ============================================================

V3_DATASET_DIR = "si_simulated_dataset_v3"
V3_MODEL_BUNDLE_NAME = "two_stage_model_bundle.pkl"
V3_MODEL_RELATIVE_DIR = "outputs/model_training_v3_two_stage"

V4_DATASET_DIR = "si_simulated_dataset_v4_pn06_lot_aware"
V4_DB_NAME = "si_simulated_dataset_v4_pn06_lot_aware.sqlite"

# Source options: "auto", "sqlite", "csv"
V4_SOURCE = "auto"

# Leave None to use V4_DATASET_DIR/outputs/pn06_v4_analysis_with_v3_models_v5_background_aware
OUTPUT_DIR = None

# If True and V4 SQLite exists, write analysis outputs back into the SQLite DB.
WRITE_OUTPUTS_TO_SQLITE = True

# Model tasks to apply. Missing models in the bundle are skipped automatically.
STAGE1_TASKS = [
    "stage1_overall_status",
    "stage1_impedance_status",
    "stage1_loss_status",
]
APPLY_STAGE2 = True

# Stage 3 rule thresholds. These are intentionally transparent and editable.
# V2 rule changes:
# - Regional issue is evaluated before fixed-test-point issue.
# - Sparse random local defects are evaluated before normal-lot fallback.
# - Raw lot-level Z / loss drift is used to catch global issues even when
#   measurement-level root-cause predictions are conservative.
NORMAL_ABNORMAL_RATE_MAX = 0.08
NORMAL_OUT_OF_SPEC_RATE_MAX = 0.03
GLOBAL_LOSS_RATE_MIN = 0.50
GLOBAL_WIDTH_RATE_MIN = 0.50
RAW_GLOBAL_LOSS_EXCESS_MEDIAN_MIN = 0.10
RAW_GLOBAL_WIDTH_ABS_Z_MEAN_DEV_MEDIAN_MIN = 0.008
RAW_GLOBAL_WIDTH_DIRECTION_CONSISTENCY_MIN = 0.60
SINGLE_BOARD_RATE_MIN = 0.60
SINGLE_BOARD_CONCENTRATION_MIN = 1.8
REGIONAL_RATE_MIN = 0.50
REGIONAL_CONCENTRATION_MIN = 2.0
FIXED_TEST_POINT_REPEAT_RATE_MIN = 0.60
FIXED_TEST_POINT_REGION_RATE_MAX = 0.50
RANDOM_LOCAL_DEFECT_RATE_MIN = 0.02
RANDOM_LOCAL_DEFECT_RATE_MAX = 0.20
RANDOM_LOCAL_ABNORMAL_RATE_MAX = 0.35
RANDOM_LOCAL_BOARD_COVERAGE_MIN = 0.25
RANDOM_LOCAL_REGION_COVERAGE_MIN = 0.50

# Point-level signal-aware thresholds.  These keep weak/systematic shifts from
# being over-interpreted as exact root causes at a single measurement point.
POINT_CONFIDENCE_MIN = 0.55
POINT_CONFIDENT_LOSS_EXCESS_RATIO_MIN = 0.10
POINT_WEAK_LOSS_EXCESS_RATIO_MIN = 0.05
POINT_CONFIDENT_IMPEDANCE_ABS_Z_DEV_PCT_MIN = 0.030
POINT_WEAK_IMPEDANCE_ABS_Z_DEV_PCT_MIN = 0.015
POINT_STRONG_LOCAL_TDR_ABS_PCT_MIN = 0.070

# Stage-3 uses signal rates rather than exact Stage-2 root-cause rates.
GLOBAL_LOSS_SIGNAL_RATE_MIN = 0.50
GLOBAL_WIDTH_SIGNAL_RATE_MIN = 0.50
NORMAL_CONFIDENT_ABNORMAL_RATE_MAX = 0.08
NORMAL_SIGNAL_ABNORMAL_RATE_MAX = 0.25

# Secondary signal thresholds. These do not replace the primary lot diagnosis;
# they surface smaller co-existing signals for engineering review.
SECONDARY_LOSS_SIGNAL_MIN = 0.05
SECONDARY_WIDTH_SIGNAL_MIN = 0.05
SECONDARY_LOCAL_SIGNAL_MIN = 0.01
SECONDARY_VIA_SIGNAL_MIN = 0.01
SECONDARY_RETURN_PATH_SIGNAL_MIN = 0.01
SECONDARY_RAW_LOSS_EXCESS_MIN = 0.05
SECONDARY_RAW_WIDTH_ABS_Z_DEV_MIN = 0.006

# Background weak-signal handling. Normal lots often contain a few weak loss/width
# signals due to measurement/model noise or normal process variation. These should
# be surfaced as background_weak_signal for review, but should not change the final
# diagnosis into *_with_secondary_signal.
BACKGROUND_CONFIDENT_ABNORMAL_RATE_MAX = 0.03
BACKGROUND_SIGNAL_ABNORMAL_RATE_MAX = 0.18
BACKGROUND_OUT_OF_SPEC_RATE_MAX = 0.03
NORMAL_SECONDARY_LOCAL_MIN = 0.03
NORMAL_SECONDARY_VIA_MIN = 0.02
NORMAL_SECONDARY_RETURN_PATH_MIN = 0.02


# ============================================================
# Utility helpers
# ============================================================

def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
        (table,),
    ).fetchone()
    return row is not None


def view_exists(conn: sqlite3.Connection, view: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='view' AND name=?;",
        (view,),
    ).fetchone()
    return row is not None


def get_table_columns(conn: sqlite3.Connection, table_or_view: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({quote_ident(table_or_view)});").fetchall()
    return [r[1] for r in rows]


def resolve_model_bundle_path() -> Path:
    """Find the V3 two-stage model bundle."""
    preferred = Path(V3_DATASET_DIR) / V3_MODEL_RELATIVE_DIR / V3_MODEL_BUNDLE_NAME
    if preferred.exists():
        return preferred

    # Also try relative to the current working directory and nearby folders.
    candidates = []
    candidates.extend(Path(".").glob(f"**/{V3_MODEL_BUNDLE_NAME}"))
    candidates.extend(Path(V3_DATASET_DIR).glob(f"**/{V3_MODEL_BUNDLE_NAME}")) if Path(V3_DATASET_DIR).exists() else []

    unique = []
    seen = set()
    for p in candidates:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if rp not in seen and p.exists():
            seen.add(rp)
            unique.append(p)

    if unique:
        # Prefer paths containing model_training_v3_two_stage.
        for p in unique:
            if "model_training_v3_two_stage" in str(p):
                return p
        return unique[0]

    raise FileNotFoundError(
        "Cannot find V3 two-stage model bundle.\n"
        f"Expected: {preferred}\n"
        "Please run train_diagnostic_models.py first."
    )


def resolve_v4_dataset_dir() -> Path:
    d = Path(V4_DATASET_DIR)
    if d.exists():
        return d
    fallbacks = ["si_simulated_dataset_v4_pn06_lot_aware", "si_v4_test"]
    for name in fallbacks:
        p = Path(name)
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot find V4 PN06 lot-aware dataset directory.\n"
        f"Expected: {V4_DATASET_DIR}\n"
        "Please run generate_validation_lot.py first."
    )


def resolve_v4_source(dataset_dir: Path) -> Tuple[str, Optional[Path]]:
    db_path = dataset_dir / V4_DB_NAME
    if V4_SOURCE in {"auto", "sqlite"} and db_path.exists():
        return "sqlite", db_path
    if V4_SOURCE == "sqlite":
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    csv_path = dataset_dir / "processed" / "si_fingerprint_features.csv"
    if csv_path.exists():
        return "csv", None

    raise FileNotFoundError(
        "Cannot find V4 data source. Tried SQLite and processed/si_fingerprint_features.csv."
    )


def load_v4_features(dataset_dir: Path) -> Tuple[pd.DataFrame, str, Optional[Path]]:
    source, db_path = resolve_v4_source(dataset_dir)
    if source == "sqlite":
        assert db_path is not None
        with sqlite3.connect(db_path) as conn:
            # Use the full feature table, because it contains all model feature columns.
            if not table_exists(conn, "si_fingerprint_features"):
                raise ValueError(f"DB missing table si_fingerprint_features: {db_path}")
            df = pd.read_sql_query("SELECT * FROM si_fingerprint_features", conn)
        return df, f"sqlite: {db_path}", db_path

    csv_path = dataset_dir / "processed" / "si_fingerprint_features.csv"
    df = pd.read_csv(csv_path)
    return df, f"csv: {csv_path}", None


def load_v3_bundle(bundle_path: Path) -> Dict:
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    if not isinstance(bundle, dict):
        raise ValueError(f"Invalid model bundle format: {bundle_path}")
    return bundle


def clean_label_token(x: object) -> str:
    text = str(x)
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() or ch in {"_", "-"} else "_")
    return "".join(out)


# ============================================================
# Prediction helpers
# ============================================================

def ensure_feature_columns(df: pd.DataFrame, feature_cols: Iterable[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Return X with exactly feature_cols. Missing columns are added as NaN."""
    x = df.copy()
    missing = []
    for col in feature_cols:
        if col not in x.columns:
            x[col] = np.nan
            missing.append(col)
    for col in feature_cols:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    return x[list(feature_cols)], missing


def model_classes(model) -> Optional[np.ndarray]:
    if hasattr(model, "classes_"):
        return getattr(model, "classes_")
    # Pipeline usually exposes classes_, but keep fallback for robustness.
    try:
        return model.named_steps["model"].classes_
    except Exception:
        return None


def apply_one_model(
    df: pd.DataFrame,
    model_info: Dict,
    pred_col: str,
    prob_prefix: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply one bundled sklearn model and add prediction/probability columns."""
    if not model_info or "model" not in model_info or "feature_cols" not in model_info:
        raise ValueError(f"Invalid model_info for {pred_col}")

    model = model_info["model"]
    feature_cols = list(model_info["feature_cols"])
    X, missing = ensure_feature_columns(df, feature_cols)

    pred = model.predict(X)
    out = pd.DataFrame(index=df.index)
    out[pred_col] = pred

    # Confidence and class probabilities when available.
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)
            classes = model_classes(model)
            if classes is not None and probs.ndim == 2:
                out[f"{prob_prefix}_confidence"] = np.max(probs, axis=1)
                for j, cls in enumerate(classes):
                    out[f"{prob_prefix}_prob_{clean_label_token(cls)}"] = probs[:, j]
        except Exception as exc:
            print(f"[WARN] predict_proba failed for {pred_col}: {exc}")

    missing_df = pd.DataFrame({
        "prediction_task": [pred_col],
        "model_name": [model_info.get("best_model_name", "unknown")],
        "feature_count": [len(feature_cols)],
        "missing_feature_count": [len(missing)],
        "missing_features": [";".join(missing)],
    })
    return out, missing_df


def root_group(label: object) -> str:
    label = str(label)
    if label in {"golden_normal", "normal_production", "normal"}:
        return "normal"
    if label in {"roughness_high", "high_df"}:
        return "loss_related"
    if label == "width_variation":
        return "impedance_width"
    if label == "local_defect":
        return "local_defect"
    if label == "via_stub":
        return "via_stub"
    if label == "return_path_issue":
        return "return_path_issue"
    return "unknown"


def status_score(status: object) -> int:
    mapping = {"normal": 0, "warning": 1, "out_of_spec": 2, "severe": 3}
    return mapping.get(str(status), 0)


def _row_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        value = row.get(col, default)
        value = float(value)
        if np.isnan(value):
            return default
        return value
    except Exception:
        return default


def _row_status_score(row: pd.Series, col: str) -> int:
    return status_score(row.get(col, "normal"))


def signal_group_from_signal_type(signal_type: object) -> str:
    """Collapse point signal type into the group used by Stage-3 aggregation."""
    s = str(signal_type)
    if s in {"loss_related", "weak_loss_shift"}:
        return "loss_related"
    if s in {"impedance_width_shift", "weak_impedance_shift"}:
        return "impedance_width"
    if s == "local_defect":
        return "local_defect"
    if s == "via_stub":
        return "via_stub"
    if s == "return_path_issue":
        return "return_path_issue"
    if s == "mixed_signal":
        return "mixed_signal"
    return "normal"


def true_label_to_signal_group(label: object) -> str:
    """Ground-truth grouping for evaluating coarse point-level signal quality."""
    return root_group(label)


def derive_point_signal_type(row: pd.Series) -> str:
    """Create a signal-aware point label.

    The goal is not to force an exact root cause from one measurement point.
    Clear local/via/return-path fingerprints remain point-level diagnoses.
    Weak loss/impedance shifts are kept as weak signals so Stage 3 can decide
    whether they are systematic across the lot.
    """
    exact_group = str(row.get("pred_stage2_root_group", "unknown"))
    conf = _row_float(row, "pred_stage2_root_cause_confidence", 0.0)

    loss_score = _row_status_score(row, "pred_stage1_loss_status")
    imp_score = _row_status_score(row, "pred_stage1_impedance_status")
    overall_score = _row_status_score(row, "pred_stage1_overall_status")

    raw_loss = _row_float(row, "loss_baseline_excess_ratio_max", 0.0)
    z_abs = abs(_row_float(row, "Z_mean_target_dev_pct", 0.0))
    tdr_abs_pct = _row_float(row, "TDR_peak_abs_pct", 0.0)
    regions_above_spec = _row_float(row, "TDR_num_regions_above_7pct", 0.0)

    # Strong, shape-specific point issues: keep these at point-level.
    if exact_group in {"local_defect", "via_stub", "return_path_issue"}:
        if conf >= 0.35 or regions_above_spec >= 1 or tdr_abs_pct >= POINT_STRONG_LOCAL_TDR_ABS_PCT_MIN:
            return exact_group

    loss_pred = exact_group == "loss_related"
    imp_pred = exact_group == "impedance_width"

    loss_confident = (
        loss_score >= 2
        or raw_loss >= POINT_CONFIDENT_LOSS_EXCESS_RATIO_MIN
        or (loss_pred and conf >= POINT_CONFIDENCE_MIN)
    )
    imp_confident = (
        imp_score >= 2
        or z_abs >= POINT_CONFIDENT_IMPEDANCE_ABS_Z_DEV_PCT_MIN
        or (imp_pred and conf >= POINT_CONFIDENCE_MIN)
    )

    loss_weak = (
        loss_score >= 1
        or raw_loss >= POINT_WEAK_LOSS_EXCESS_RATIO_MIN
        or loss_pred
    )
    imp_weak = (
        imp_score >= 1
        or z_abs >= POINT_WEAK_IMPEDANCE_ABS_Z_DEV_PCT_MIN
        or imp_pred
    )

    if loss_confident and imp_confident:
        return "mixed_signal"
    if loss_confident:
        return "loss_related"
    if imp_confident:
        return "impedance_width_shift"
    if loss_weak and imp_weak:
        # Leave weak mixed behavior to the lot-level aggregation.
        return "normal_or_weak"
    if loss_weak:
        return "weak_loss_shift"
    if imp_weak:
        return "weak_impedance_shift"

    # If Stage 1 overall says warning but neither loss nor impedance explains it,
    # keep it weak rather than inventing a point-level root cause.
    if overall_score >= 1:
        return "normal_or_weak"
    return "normal_or_weak"


def apply_v3_models_to_v4(df: pd.DataFrame, bundle: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pred_df = df.copy()
    missing_reports = []

    stage1_models = bundle.get("stage1_models", {}) or {}
    for task in STAGE1_TASKS:
        if task not in stage1_models:
            print(f"[WARN] Stage-1 model missing in bundle: {task}")
            continue
        pred_col = f"pred_{task}"
        prob_prefix = f"pred_{task}"
        print(f"[PREDICT] {task}")
        out, missing_df = apply_one_model(pred_df, stage1_models[task], pred_col, prob_prefix)
        pred_df = pd.concat([pred_df, out], axis=1)
        missing_reports.append(missing_df)

    if APPLY_STAGE2 and bundle.get("stage2_model") is not None:
        print("[PREDICT] stage2_root_cause")
        out, missing_df = apply_one_model(
            pred_df,
            bundle["stage2_model"],
            "pred_stage2_root_cause",
            "pred_stage2_root_cause",
        )
        pred_df = pd.concat([pred_df, out], axis=1)
        missing_reports.append(missing_df)
    else:
        print("[WARN] Stage-2 model missing or disabled.")

    # Derived prediction fields.
    if "pred_stage2_root_cause" in pred_df.columns:
        pred_df["pred_stage2_root_group_exact"] = pred_df["pred_stage2_root_cause"].map(root_group)
        pred_df["pred_stage2_root_group"] = pred_df["pred_stage2_root_group_exact"]
        pred_df["pred_stage2_abnormal_flag"] = (pred_df["pred_stage2_root_group_exact"] != "normal").astype(int)
    else:
        pred_df["pred_stage2_root_group_exact"] = "unknown"
        pred_df["pred_stage2_root_group"] = "unknown"
        pred_df["pred_stage2_abnormal_flag"] = 0

    if "pred_stage1_overall_status" in pred_df.columns:
        pred_df["pred_stage1_overall_score"] = pred_df["pred_stage1_overall_status"].map(status_score)
        pred_df["pred_stage1_warning_or_worse_flag"] = (pred_df["pred_stage1_overall_score"] >= 1).astype(int)
        pred_df["pred_stage1_out_of_spec_or_worse_flag"] = (pred_df["pred_stage1_overall_score"] >= 2).astype(int)
        pred_df["pred_stage1_severe_flag"] = (pred_df["pred_stage1_overall_score"] >= 3).astype(int)
    else:
        pred_df["pred_stage1_overall_score"] = 0
        pred_df["pred_stage1_warning_or_worse_flag"] = 0
        pred_df["pred_stage1_out_of_spec_or_worse_flag"] = 0
        pred_df["pred_stage1_severe_flag"] = 0

    # Legacy exact abnormal flag: useful for auditing, but not the final Stage-3 signal.
    pred_df["pred_any_abnormal_flag_exact"] = (
        (pred_df["pred_stage1_warning_or_worse_flag"] == 1)
        | (pred_df["pred_stage2_abnormal_flag"] == 1)
    ).astype(int)

    # Signal-aware point labels.  Weak shifts are intentionally not forced into exact root causes.
    pred_df["pred_point_signal_type"] = pred_df.apply(derive_point_signal_type, axis=1)
    pred_df["pred_point_signal_group"] = pred_df["pred_point_signal_type"].map(signal_group_from_signal_type)
    pred_df["pred_point_weak_signal_flag"] = pred_df["pred_point_signal_type"].isin(
        ["weak_loss_shift", "weak_impedance_shift", "normal_or_weak"]
    ).astype(int)
    pred_df["pred_point_signal_abnormal_flag"] = (pred_df["pred_point_signal_group"] != "normal").astype(int)
    pred_df["pred_point_confident_abnormal_flag"] = pred_df["pred_point_signal_type"].isin(
        ["loss_related", "impedance_width_shift", "local_defect", "via_stub", "return_path_issue", "mixed_signal"]
    ).astype(int)

    # Keep the old column name for compatibility, but make it signal-aware.
    pred_df["pred_any_abnormal_flag"] = pred_df["pred_point_signal_abnormal_flag"]

    missing_all = pd.concat(missing_reports, ignore_index=True) if missing_reports else pd.DataFrame()
    return pred_df, missing_all


# ============================================================
# Aggregation and Stage 3 diagnosis
# ============================================================

def safe_rate(series: pd.Series, value: object) -> float:
    if len(series) == 0:
        return 0.0
    return float((series == value).mean())


def summarize_group(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row["total_measurements"] = int(len(g))

        # Legacy exact-model abnormal rate, retained for audit.
        if "pred_any_abnormal_flag_exact" in g.columns:
            row["pred_exact_any_abnormal_rate"] = float(g["pred_any_abnormal_flag_exact"].mean())
        else:
            row["pred_exact_any_abnormal_rate"] = float(g["pred_any_abnormal_flag"].mean())

        # Signal-aware rates used by Stage 3.
        row["pred_any_abnormal_rate"] = float(g["pred_point_signal_abnormal_flag"].mean())
        row["pred_confident_abnormal_rate"] = float(g["pred_point_confident_abnormal_flag"].mean())
        row["pred_weak_signal_rate"] = float(g["pred_point_weak_signal_flag"].mean())

        row["pred_stage1_warning_or_worse_rate"] = float(g["pred_stage1_warning_or_worse_flag"].mean())
        row["pred_stage1_out_of_spec_or_worse_rate"] = float(g["pred_stage1_out_of_spec_or_worse_flag"].mean())
        row["pred_stage1_severe_rate"] = float(g["pred_stage1_severe_flag"].mean())

        # Exact Stage-2 rates for audit.
        row["pred_stage2_exact_loss_related_rate"] = safe_rate(g["pred_stage2_root_group_exact"], "loss_related")
        row["pred_stage2_exact_impedance_width_rate"] = safe_rate(g["pred_stage2_root_group_exact"], "impedance_width")
        row["pred_stage2_exact_local_defect_rate"] = safe_rate(g["pred_stage2_root_group_exact"], "local_defect")
        row["pred_stage2_exact_via_stub_rate"] = safe_rate(g["pred_stage2_root_group_exact"], "via_stub")
        row["pred_stage2_exact_return_path_rate"] = safe_rate(g["pred_stage2_root_group_exact"], "return_path_issue")

        # Signal-aware rates for lot-level diagnosis.
        row["pred_loss_related_rate"] = safe_rate(g["pred_point_signal_group"], "loss_related")
        row["pred_impedance_width_rate"] = safe_rate(g["pred_point_signal_group"], "impedance_width")
        row["pred_local_defect_rate"] = safe_rate(g["pred_point_signal_group"], "local_defect")
        row["pred_via_stub_rate"] = safe_rate(g["pred_point_signal_group"], "via_stub")
        row["pred_return_path_rate"] = safe_rate(g["pred_point_signal_group"], "return_path_issue")
        row["pred_mixed_signal_rate"] = safe_rate(g["pred_point_signal_group"], "mixed_signal")
        row["pred_weak_loss_shift_rate"] = safe_rate(g["pred_point_signal_type"], "weak_loss_shift")
        row["pred_weak_impedance_shift_rate"] = safe_rate(g["pred_point_signal_type"], "weak_impedance_shift")

        # Optional true labels if V4 simulation metadata exists.
        if "true_lot_pattern" in g.columns:
            row["true_lot_pattern"] = str(g["true_lot_pattern"].mode(dropna=True).iloc[0]) if not g["true_lot_pattern"].mode(dropna=True).empty else ""
        if "label" in g.columns:
            row["true_abnormal_rate"] = float((~g["label"].isin(["golden_normal", "normal_production"])).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def true_lot_pattern_to_group(pattern: object) -> str:
    p = str(pattern)
    if p == "normal_lot":
        return "normal_lot"
    if p in {"lot_global_loss_roughness", "lot_global_loss_high_df"}:
        return "lot_global_loss_issue"
    if p == "lot_global_impedance_width_shift":
        return "lot_global_impedance_width_issue"
    if p == "single_board_issue":
        return "single_board_issue"
    if p == "regional_issue":
        return "regional_issue"
    if p == "fixed_test_point_structure_issue":
        return "fixed_test_point_structure_issue"
    if p == "random_local_defect":
        return "random_local_defect"
    if p == "mixed_issue":
        return "mixed_issue"
    return "unknown"


def rule_based_lot_diagnosis(row: pd.Series) -> str:
    """Transparent signal-aware Stage-3 PN06 lot diagnosis rule.

    Design intent:
    - Single-point exact root cause is treated as a noisy sensor.
    - Weak loss/impedance shifts are not forced into exact point diagnoses.
    - If weak shifts repeat across the lot, Stage 3 upgrades them to global issues.
    - Clear local/via/return-path signatures still count as confident point-level issues.
    """
    signal_a = float(row.get("pred_any_abnormal_rate", 0.0))
    confident_a = float(row.get("pred_confident_abnormal_rate", 0.0))
    outspec = float(row.get("pred_stage1_out_of_spec_or_worse_rate", 0.0))

    loss = float(row.get("pred_loss_related_rate", 0.0))
    width = float(row.get("pred_impedance_width_rate", 0.0))
    local = float(row.get("pred_local_defect_rate", 0.0))
    via = float(row.get("pred_via_stub_rate", 0.0))
    ret = float(row.get("pred_return_path_rate", 0.0))
    mixed_signal = float(row.get("pred_mixed_signal_rate", 0.0))

    weak_loss = float(row.get("pred_weak_loss_shift_rate", 0.0))
    weak_width = float(row.get("pred_weak_impedance_shift_rate", 0.0))
    loss_signal = loss + weak_loss
    width_signal = width + weak_width

    max_board = float(row.get("max_board_abnormal_rate", 0.0))
    max_region = float(row.get("max_region_abnormal_rate", 0.0))
    fixed_repeat = float(row.get("fixed_test_point_repeat_rate", 0.0))

    board_conc = float(row.get("board_concentration_score", 0.0))
    region_conc = float(row.get("regional_concentration_score", 0.0))
    local_board_cov = float(row.get("local_defect_board_coverage_rate", 0.0))
    local_region_cov = float(row.get("local_defect_region_coverage_rate", 0.0))

    raw_loss_median = float(row.get("lot_median_loss_baseline_excess_ratio_max", 0.0))
    raw_width_abs_median = float(row.get("lot_median_abs_Z_mean_target_dev_pct", 0.0))
    raw_z_direction_consistency = float(row.get("lot_Z_shift_direction_consistency", 0.0))

    # 1) Global loss: either confident loss predictions, many weak loss shifts,
    # or raw Delta-L/loss excess across the lot.
    if (
        loss_signal >= GLOBAL_LOSS_SIGNAL_RATE_MIN
        or raw_loss_median >= RAW_GLOBAL_LOSS_EXCESS_MEDIAN_MIN
    ):
        if loss_signal >= max(width_signal, local, via, ret) or raw_loss_median >= RAW_GLOBAL_LOSS_EXCESS_MEDIAN_MIN:
            return "lot_global_loss_issue"

    # 2) Global impedance/width: either many width-like/weak-width signals,
    # or a same-direction Z_mean shift across the lot.
    if (
        width_signal >= GLOBAL_WIDTH_SIGNAL_RATE_MIN
        or (
            raw_width_abs_median >= RAW_GLOBAL_WIDTH_ABS_Z_MEAN_DEV_MEDIAN_MIN
            and raw_z_direction_consistency >= RAW_GLOBAL_WIDTH_DIRECTION_CONSISTENCY_MIN
        )
    ):
        if width_signal >= max(loss_signal, local, via, ret) or raw_width_abs_median >= RAW_GLOBAL_WIDTH_ABS_Z_MEAN_DEV_MEDIAN_MIN:
            return "lot_global_impedance_width_issue"

    # 3) Single-board issue: one sampled board dominates confident abnormality.
    if max_board >= SINGLE_BOARD_RATE_MIN and (confident_a < 0.50 or board_conc >= SINGLE_BOARD_CONCENTRATION_MIN):
        return "single_board_issue"

    # 4) Regional issue before fixed test point. A regional issue naturally makes
    # repeated map positions abnormal across boards.
    if max_region >= REGIONAL_RATE_MIN and region_conc >= REGIONAL_CONCENTRATION_MIN:
        return "regional_issue"

    # 5) Fixed test-point / structure issue: repeated TP across boards, not
    # primarily explained by a concentrated region.
    if (
        fixed_repeat >= FIXED_TEST_POINT_REPEAT_RATE_MIN
        and max_region < FIXED_TEST_POINT_REGION_RATE_MAX
        and confident_a < 0.50
    ):
        return "fixed_test_point_structure_issue"

    # 6) Sparse random local defects: low total confident rate, local-defect
    # predictions scattered across multiple boards/regions.
    if (
        RANDOM_LOCAL_DEFECT_RATE_MIN <= local <= RANDOM_LOCAL_DEFECT_RATE_MAX
        and confident_a <= RANDOM_LOCAL_ABNORMAL_RATE_MAX
        and local_board_cov >= RANDOM_LOCAL_BOARD_COVERAGE_MIN
        and local_region_cov >= RANDOM_LOCAL_REGION_COVERAGE_MIN
    ):
        return "random_local_defect"

    # 7) Normal lot: allow some weak point signals, but not many confident
    # abnormalities or out-of-spec measurements.
    if (
        confident_a <= NORMAL_CONFIDENT_ABNORMAL_RATE_MAX
        and signal_a <= NORMAL_SIGNAL_ABNORMAL_RATE_MAX
        and outspec <= NORMAL_OUT_OF_SPEC_RATE_MAX
    ):
        return "normal_lot"

    # If more than one signal source is visible but no single systematic pattern
    # dominates, keep it as mixed.
    if mixed_signal > 0.0 or (loss_signal > 0.10 and width_signal > 0.10) or (local > 0.02 and (loss_signal > 0.10 or width_signal > 0.10)):
        return "mixed_issue"

    return "mixed_issue"



def _is_primary_category(primary: str, category: str) -> bool:
    """Return True when a secondary category is already explained by primary."""
    if category == "loss" and primary == "lot_global_loss_issue":
        return True
    if category == "width" and primary == "lot_global_impedance_width_issue":
        return True
    if category == "local" and primary == "random_local_defect":
        return True
    if category == "via" and primary == "fixed_test_point_structure_issue":
        return True
    if category == "return_path" and primary in {"fixed_test_point_structure_issue", "regional_issue"}:
        return True
    if primary == "mixed_issue":
        return True
    return False


def secondary_lot_signals(row: pd.Series, primary: str) -> str:
    """Surface smaller co-existing lot-level evidence without changing primary diagnosis.

    V5 background-aware change:
    - For normal lots, weak loss/width shifts are usually background process
      variation rather than a true secondary issue. They are reported as
      background_weak_signal and do not change the final lot diagnosis.
    - For abnormal primary lots, secondary signals still report minor co-existing
      local/via/return-path/loss/width evidence for engineering review.
    """
    loss = float(row.get("pred_loss_related_rate", 0.0))
    width = float(row.get("pred_impedance_width_rate", 0.0))
    local = float(row.get("pred_local_defect_rate", 0.0))
    via = float(row.get("pred_via_stub_rate", 0.0))
    ret = float(row.get("pred_return_path_rate", 0.0))
    weak_loss = float(row.get("pred_weak_loss_shift_rate", 0.0))
    weak_width = float(row.get("pred_weak_impedance_shift_rate", 0.0))
    raw_loss_median = float(row.get("lot_median_loss_baseline_excess_ratio_max", 0.0))
    raw_width_abs_median = float(row.get("lot_median_abs_Z_mean_target_dev_pct", 0.0))
    raw_z_direction_consistency = float(row.get("lot_Z_shift_direction_consistency", 0.0))
    confident_a = float(row.get("pred_confident_abnormal_rate", 0.0))
    signal_a = float(row.get("pred_any_abnormal_rate", 0.0))
    outspec = float(row.get("pred_stage1_out_of_spec_or_worse_rate", 0.0))

    loss_signal = loss + weak_loss
    width_signal = width + weak_width

    # Normal lots: avoid presenting low-level weak loss/width drift as a
    # secondary issue. Keep it as background for dashboard transparency.
    if primary == "normal_lot":
        has_background = (
            loss_signal >= SECONDARY_LOSS_SIGNAL_MIN
            or width_signal >= SECONDARY_WIDTH_SIGNAL_MIN
            or raw_loss_median >= SECONDARY_RAW_LOSS_EXCESS_MIN
            or (
                raw_width_abs_median >= SECONDARY_RAW_WIDTH_ABS_Z_DEV_MIN
                and raw_z_direction_consistency >= RAW_GLOBAL_WIDTH_DIRECTION_CONSISTENCY_MIN
            )
        )

        strong_secondary_signals: List[Tuple[str, float]] = []
        if local >= NORMAL_SECONDARY_LOCAL_MIN:
            strong_secondary_signals.append(("secondary_local_defect", local))
        if via >= NORMAL_SECONDARY_VIA_MIN:
            strong_secondary_signals.append(("secondary_via_stub", via))
        if ret >= NORMAL_SECONDARY_RETURN_PATH_MIN:
            strong_secondary_signals.append(("secondary_return_path", ret))

        if strong_secondary_signals:
            strong_secondary_signals = sorted(strong_secondary_signals, key=lambda x: x[1], reverse=True)
            return ";".join(name for name, _ in strong_secondary_signals[:2])

        if (
            has_background
            and confident_a <= BACKGROUND_CONFIDENT_ABNORMAL_RATE_MAX
            and signal_a <= BACKGROUND_SIGNAL_ABNORMAL_RATE_MAX
            and outspec <= BACKGROUND_OUT_OF_SPEC_RATE_MAX
        ):
            return "background_weak_signal"

        return "none"

    signals: List[Tuple[str, float]] = []

    if not _is_primary_category(primary, "loss"):
        score = max(loss_signal, raw_loss_median)
        if loss_signal >= SECONDARY_LOSS_SIGNAL_MIN or raw_loss_median >= SECONDARY_RAW_LOSS_EXCESS_MIN:
            signals.append(("secondary_loss_shift", float(score)))

    if not _is_primary_category(primary, "width"):
        score = max(width_signal, raw_width_abs_median)
        if (
            width_signal >= SECONDARY_WIDTH_SIGNAL_MIN
            or (
                raw_width_abs_median >= SECONDARY_RAW_WIDTH_ABS_Z_DEV_MIN
                and raw_z_direction_consistency >= RAW_GLOBAL_WIDTH_DIRECTION_CONSISTENCY_MIN
            )
        ):
            signals.append(("secondary_impedance_width_shift", float(score)))

    if not _is_primary_category(primary, "local") and local >= SECONDARY_LOCAL_SIGNAL_MIN:
        signals.append(("secondary_local_defect", float(local)))

    if not _is_primary_category(primary, "via") and via >= SECONDARY_VIA_SIGNAL_MIN:
        signals.append(("secondary_via_stub", float(via)))

    if not _is_primary_category(primary, "return_path") and ret >= SECONDARY_RETURN_PATH_SIGNAL_MIN:
        signals.append(("secondary_return_path", float(ret)))

    if not signals:
        return "none"

    signals = sorted(signals, key=lambda x: x[1], reverse=True)
    # Keep the output readable. Two secondary signals are enough for a dashboard.
    return ";".join(name for name, _ in signals[:2])


def secondary_signal_type(secondary: str) -> str:
    """Classify the secondary signal so dashboards can separate background from issues."""
    if not secondary or secondary == "none":
        return "none"
    if secondary == "background_weak_signal":
        return "background_weak_signal"
    return "secondary_issue_signal"


def final_lot_diagnosis(primary: str, secondary: str) -> str:
    """Human-readable final diagnosis with secondary evidence preserved."""
    if secondary and secondary not in {"none", "background_weak_signal"} and primary != "mixed_issue":
        return f"{primary}_with_secondary_signal"
    return primary


def lot_engineering_match(row: pd.Series) -> int:
    """Engineering-oriented match.

    Exact label accuracy is kept separately. This match gives credit when a
    simulated mixed lot is diagnosed as a strong primary systematic issue but
    also reports secondary evidence, because this is often how an engineer would
    write the finding: primary cause + secondary signal.
    """
    true_group = str(row.get("true_lot_diagnosis_group", ""))
    primary = str(row.get("primary_lot_diagnosis", row.get("predicted_lot_diagnosis", "")))
    secondary = str(row.get("secondary_lot_signal", "none"))

    if primary == true_group:
        return 1

    if true_group == "mixed_issue" and primary not in {"normal_lot", "unknown"} and secondary not in {"none", "background_weak_signal"}:
        return 1

    if true_group == "random_local_defect" and (primary == "mixed_issue" or "secondary_local_defect" in secondary):
        return 1

    return 0



def _numeric_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def add_raw_lot_statistics(lot: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """Add raw lot-level statistics not dependent on model predictions.

    These statistics catch systematic PN06 lot drift that can be missed when
    a measurement-level root-cause classifier is conservative near boundaries.
    """
    rows = []
    for lot_id, g in pred_df.groupby("lot_id", dropna=False):
        row = {"lot_id": lot_id}

        # Impedance drift statistics.
        if "Z_mean_target_dev_pct" in g.columns:
            z = _numeric_col(g, "Z_mean_target_dev_pct")
            row["lot_median_Z_mean_target_dev_pct"] = float(np.nanmedian(z))
            row["lot_mean_Z_mean_target_dev_pct"] = float(np.nanmean(z))
            row["lot_median_abs_Z_mean_target_dev_pct"] = float(np.nanmedian(np.abs(z)))
            # Direction consistency: 1 means nearly all points shift in the same direction.
            valid = z.dropna()
            if len(valid) > 0:
                signs = np.sign(valid)
                signs = signs[signs != 0]
                row["lot_Z_shift_direction_consistency"] = float(abs(np.nanmean(signs))) if len(signs) else 0.0
            else:
                row["lot_Z_shift_direction_consistency"] = 0.0
        if "Z_max_dev_pct" in g.columns:
            zmax = _numeric_col(g, "Z_max_dev_pct")
            row["lot_median_Z_max_dev_pct"] = float(np.nanmedian(zmax))
            row["lot_p90_Z_max_dev_pct"] = float(np.nanpercentile(zmax, 90))
        if "TDR_peak_abs_pct" in g.columns:
            tdr = _numeric_col(g, "TDR_peak_abs_pct")
            row["lot_median_TDR_peak_abs_pct"] = float(np.nanmedian(tdr))
            row["lot_p90_TDR_peak_abs_pct"] = float(np.nanpercentile(tdr, 90))

        # Loss drift statistics.
        if "loss_baseline_excess_ratio_max" in g.columns:
            loss_excess = _numeric_col(g, "loss_baseline_excess_ratio_max")
            row["lot_median_loss_baseline_excess_ratio_max"] = float(np.nanmedian(loss_excess))
            row["lot_mean_loss_baseline_excess_ratio_max"] = float(np.nanmean(loss_excess))
            row["lot_p90_loss_baseline_excess_ratio_max"] = float(np.nanpercentile(loss_excess, 90))
        for col in [
            "DeltaL_28GHz_dB_per_in_pn_excess_ratio",
            "DeltaL_40GHz_dB_per_in_pn_excess_ratio",
            "IL_28GHz_pn_robust_z",
            "IL_40GHz_pn_robust_z",
        ]:
            if col in g.columns:
                values = _numeric_col(g, col)
                row[f"lot_median_{col}"] = float(np.nanmedian(values))
                row[f"lot_p90_{col}"] = float(np.nanpercentile(values, 90))

        rows.append(row)

    raw_stats = pd.DataFrame(rows)
    return lot.merge(raw_stats, on="lot_id", how="left")

def build_stage3_aggregations(pred_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    required = {"lot_id", "board_id", "test_point_id", "region_id"}
    missing = [c for c in required if c not in pred_df.columns]
    if missing:
        raise ValueError(
            "V4 hierarchy columns are missing. Cannot build Stage 3 aggregations. "
            f"Missing: {missing}"
        )

    board = summarize_group(pred_df, ["lot_id", "board_id"])
    region = summarize_group(pred_df, ["lot_id", "region_id"])
    tp = summarize_group(pred_df, ["lot_id", "test_point_id"])
    lot = summarize_group(pred_df, ["lot_id"])

    # Add derived maxima/concentration scores to lot table.
    # Stage 3 concentrates on confident abnormality for board/region/TP concentration.
    # Weak signals are still used for global lot-level loss/impedance shift rates.
    board_max = board.groupby("lot_id")["pred_confident_abnormal_rate"].max().rename("max_board_abnormal_rate")
    region_max = region.groupby("lot_id")["pred_confident_abnormal_rate"].max().rename("max_region_abnormal_rate")
    tp_max = tp.groupby("lot_id")["pred_confident_abnormal_rate"].max().rename("fixed_test_point_repeat_rate")

    lot = lot.merge(board_max, on="lot_id", how="left")
    lot = lot.merge(region_max, on="lot_id", how="left")
    lot = lot.merge(tp_max, on="lot_id", how="left")

    denom = lot["pred_confident_abnormal_rate"].replace(0, np.nan)
    lot["board_concentration_score"] = lot["max_board_abnormal_rate"] / denom
    lot["regional_concentration_score"] = lot["max_region_abnormal_rate"] / denom
    lot["board_concentration_score"] = lot["board_concentration_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    lot["regional_concentration_score"] = lot["regional_concentration_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Local defect randomness: how many boards/regions have at least one predicted local defect.
    local_df = pred_df[pred_df["pred_point_signal_group"] == "local_defect"].copy()
    if local_df.empty:
        lot["local_defect_board_coverage_rate"] = 0.0
        lot["local_defect_region_coverage_rate"] = 0.0
    else:
        board_total = pred_df.groupby("lot_id")["board_id"].nunique()
        region_total = pred_df.groupby("lot_id")["region_id"].nunique()
        board_cov = local_df.groupby("lot_id")["board_id"].nunique() / board_total
        region_cov = local_df.groupby("lot_id")["region_id"].nunique() / region_total
        lot["local_defect_board_coverage_rate"] = lot["lot_id"].map(board_cov).fillna(0.0)
        lot["local_defect_region_coverage_rate"] = lot["lot_id"].map(region_cov).fillna(0.0)

    # Add raw lot-level Z/loss statistics before applying Stage-3 rules.
    lot = add_raw_lot_statistics(lot, pred_df)

    lot["primary_lot_diagnosis"] = lot.apply(rule_based_lot_diagnosis, axis=1)
    lot["secondary_lot_signal"] = lot.apply(lambda r: secondary_lot_signals(r, str(r["primary_lot_diagnosis"])), axis=1)
    lot["secondary_signal_type"] = lot["secondary_lot_signal"].map(secondary_signal_type)
    lot["final_lot_diagnosis"] = lot.apply(
        lambda r: final_lot_diagnosis(str(r["primary_lot_diagnosis"]), str(r["secondary_lot_signal"])),
        axis=1,
    )

    # Backward-compatible name used by old dashboards and metrics.
    # This remains the primary diagnosis so exact label accuracy remains comparable.
    lot["predicted_lot_diagnosis"] = lot["primary_lot_diagnosis"]

    if "true_lot_pattern" in lot.columns:
        lot["true_lot_diagnosis_group"] = lot["true_lot_pattern"].map(true_lot_pattern_to_group)
        lot["lot_diagnosis_match"] = (lot["primary_lot_diagnosis"] == lot["true_lot_diagnosis_group"]).astype(int)
        lot["lot_engineering_match"] = lot.apply(lot_engineering_match, axis=1)

    return {
        "measurement_predictions": pred_df,
        "board_prediction_summary": board,
        "region_prediction_summary": region,
        "test_point_prediction_summary": tp,
        "lot_prediction_summary": lot,
    }


# ============================================================
# Evaluation / plots
# ============================================================

def save_confusion_matrix_csv_and_png(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: List[str],
    title: str,
    csv_path: Path,
    png_path: Path,
) -> None:
    if confusion_matrix is None:
        return
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    cm_df.to_csv(csv_path, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def evaluate_outputs(outputs: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    meas = outputs["measurement_predictions"]
    lot = outputs["lot_prediction_summary"]

    metrics_rows = []

    # Measurement-level Stage 1 comparisons against V4 generated status columns, if present.
    for task in STAGE1_TASKS:
        pred_col = f"pred_{task}"
        if task in meas.columns and pred_col in meas.columns and accuracy_score is not None:
            y_true = meas[task].astype(str)
            y_pred = meas[pred_col].astype(str)
            labels = ["normal", "warning", "out_of_spec", "severe"]
            labels = [l for l in labels if l in set(y_true).union(set(y_pred))]
            acc = accuracy_score(y_true, y_pred)
            metrics_rows.append({"level": "measurement", "task": task, "accuracy": acc, "samples": len(meas)})
            save_confusion_matrix_csv_and_png(
                y_true, y_pred, labels,
                title=f"V4 measurement validation - {task}",
                csv_path=out_dir / f"confusion_matrix_measurement_{task}.csv",
                png_path=out_dir / f"confusion_matrix_measurement_{task}.png",
            )
            if classification_report is not None:
                rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
                pd.DataFrame(rep).T.to_csv(out_dir / f"classification_report_measurement_{task}.csv", encoding="utf-8-sig")

    # Measurement-level Stage 2 comparison against V4 generated point label, if present.
    if "label" in meas.columns and "pred_stage2_root_cause" in meas.columns and accuracy_score is not None:
        y_true = meas["label"].astype(str)
        y_pred = meas["pred_stage2_root_cause"].astype(str)
        labels = [
            "normal_production", "roughness_high", "high_df", "width_variation",
            "local_defect", "via_stub", "return_path_issue",
        ]
        labels = [l for l in labels if l in set(y_true).union(set(y_pred))]
        acc = accuracy_score(y_true, y_pred)
        metrics_rows.append({"level": "measurement", "task": "stage2_root_cause", "accuracy": acc, "samples": len(meas)})
        save_confusion_matrix_csv_and_png(
            y_true, y_pred, labels,
            title="V4 measurement validation - Stage 2 root cause",
            csv_path=out_dir / "confusion_matrix_measurement_stage2_root_cause.csv",
            png_path=out_dir / "confusion_matrix_measurement_stage2_root_cause.png",
        )
        if classification_report is not None:
            rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
            pd.DataFrame(rep).T.to_csv(out_dir / "classification_report_measurement_stage2_root_cause.csv", encoding="utf-8-sig")

    # Measurement-level Stage 2 coarse signal-group comparison.
    # This is often the more meaningful point-level metric because weak/systematic
    # shifts are intended to be resolved at Stage 3 rather than as exact point labels.
    if "label" in meas.columns and "pred_point_signal_group" in meas.columns and accuracy_score is not None:
        y_true = meas["label"].map(true_label_to_signal_group).astype(str)
        y_pred = meas["pred_point_signal_group"].astype(str)
        labels = ["normal", "loss_related", "impedance_width", "local_defect", "via_stub", "return_path_issue", "mixed_signal"]
        labels = [l for l in labels if l in set(y_true).union(set(y_pred))]
        acc = accuracy_score(y_true, y_pred)
        metrics_rows.append({"level": "measurement", "task": "stage2_signal_group", "accuracy": acc, "samples": len(meas)})
        save_confusion_matrix_csv_and_png(
            y_true, y_pred, labels,
            title="V4 measurement validation - Stage 2 signal group",
            csv_path=out_dir / "confusion_matrix_measurement_stage2_signal_group.csv",
            png_path=out_dir / "confusion_matrix_measurement_stage2_signal_group.png",
        )
        if classification_report is not None:
            rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
            pd.DataFrame(rep).T.to_csv(out_dir / "classification_report_measurement_stage2_signal_group.csv", encoding="utf-8-sig")

    # Lot-level Stage 3 comparison against mapped true lot pattern, if present.
    if "true_lot_diagnosis_group" in lot.columns and "primary_lot_diagnosis" in lot.columns and accuracy_score is not None:
        y_true = lot["true_lot_diagnosis_group"].astype(str)
        y_pred = lot["primary_lot_diagnosis"].astype(str)
        labels = [
            "normal_lot", "lot_global_loss_issue", "lot_global_impedance_width_issue",
            "single_board_issue", "regional_issue", "fixed_test_point_structure_issue",
            "random_local_defect", "mixed_issue",
        ]
        labels = [l for l in labels if l in set(y_true).union(set(y_pred))]
        acc = accuracy_score(y_true, y_pred)
        metrics_rows.append({"level": "lot", "task": "stage3_primary_lot_diagnosis", "accuracy": acc, "samples": len(lot)})
        # Backward-compatible metric name.
        metrics_rows.append({"level": "lot", "task": "stage3_lot_diagnosis", "accuracy": acc, "samples": len(lot)})
        save_confusion_matrix_csv_and_png(
            y_true, y_pred, labels,
            title="V4 lot-level validation - Stage 3 primary diagnosis",
            csv_path=out_dir / "confusion_matrix_lot_stage3_primary_diagnosis.csv",
            png_path=out_dir / "confusion_matrix_lot_stage3_primary_diagnosis.png",
        )

        if "lot_engineering_match" in lot.columns:
            eng_acc = float(pd.to_numeric(lot["lot_engineering_match"], errors="coerce").fillna(0).mean())
            metrics_rows.append({"level": "lot", "task": "stage3_engineering_match", "accuracy": eng_acc, "samples": len(lot)})

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(out_dir / "analysis_metrics_summary.csv", index=False, encoding="utf-8-sig")
        print("\n=== Validation metrics summary ===")
        print(metrics_df.to_string(index=False))


def write_outputs_to_sqlite(db_path: Path, outputs: Dict[str, pd.DataFrame]) -> None:
    print(f"[SQLITE] Writing prediction output tables back to: {db_path}")
    with sqlite3.connect(db_path) as conn:
        for table, df in outputs.items():
            table_name = f"v3_model_{table}"
            df.to_sql(table_name, conn, if_exists="replace", index=False)
        # Convenience views.
        conn.execute("DROP VIEW IF EXISTS view_v3_model_lot_dashboard;")
        conn.execute("""
CREATE VIEW view_v3_model_lot_dashboard AS
SELECT *
FROM v3_model_lot_prediction_summary
ORDER BY lot_id;
""")
        conn.execute("DROP VIEW IF EXISTS view_v3_model_high_risk_measurements;")
        conn.execute("""
CREATE VIEW view_v3_model_high_risk_measurements AS
SELECT *
FROM v3_model_measurement_predictions
WHERE pred_any_abnormal_flag = 1
ORDER BY lot_id, board_id, test_point_id;
""")
        conn.commit()


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=== PN06 V4 Lot-Aware Analysis Using V3 Two-Stage Models: V3 Signal-Aware Rules ===")

    v4_dir = resolve_v4_dataset_dir()
    bundle_path = resolve_model_bundle_path()
    out_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else v4_dir / "outputs" / "pn06_lot_diagnosis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"V3 model bundle : {bundle_path.resolve()}")
    print(f"V4 dataset dir  : {v4_dir.resolve()}")
    print(f"Output folder   : {out_dir.resolve()}")

    bundle = load_v3_bundle(bundle_path)
    df_v4, source_desc, db_path = load_v4_features(v4_dir)
    print(f"V4 source       : {source_desc}")
    print(f"V4 rows         : {len(df_v4):,}")
    print(f"V4 columns      : {len(df_v4.columns):,}")

    pred_df, missing_features = apply_v3_models_to_v4(df_v4, bundle)
    if not missing_features.empty:
        missing_path = out_dir / "missing_features_report.csv"
        missing_features.to_csv(missing_path, index=False, encoding="utf-8-sig")
        print(f"[SAVED] {missing_path}")
        print(missing_features[["prediction_task", "feature_count", "missing_feature_count"]].to_string(index=False))

    outputs = build_stage3_aggregations(pred_df)

    # Save CSV outputs.
    for name, table in outputs.items():
        path = out_dir / f"{name}.csv"
        table.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[SAVED] {path} rows={len(table):,}")

    evaluate_outputs(outputs, out_dir)

    if WRITE_OUTPUTS_TO_SQLITE and db_path is not None:
        write_outputs_to_sqlite(db_path, outputs)

    print("\n=== Suggested first files to inspect ===")
    print(out_dir / "lot_prediction_summary.csv")
    print(out_dir / "measurement_predictions.csv")
    print(out_dir / "analysis_metrics_summary.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
