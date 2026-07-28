"""
generate_validation_lot.py

V4 lot-aware PN06 simulation dataset generator for high-speed PCB
signal-integrity abnormality diagnosis.

Direct-run design:
- PN06 is treated as a production lot dataset.
- One lot represents 60 PNL production, with 10 sampled boards and
  20 measurement test points per sampled board.
- Each measurement point has VNA/S-parameter, TDR, Delta-L, and V3 two-stage
  features.
- Additional board/test-point/region/lot aggregation features are generated
  to support Stage 3 diagnosis:
    global lot issue / single-board issue / regional issue /
    fixed-test-point issue / random local defect / mixed issue.

Dependency:
- Put this file in the same folder as:
    generate_training_dataset.py
  because V4 reuses the V3 physics-inspired curve simulator and feature extractor.

Important:
- This is NOT a full-wave EM solver.
- The objective is to create structured, controllable, lot-aware simulation data
  for AI pipeline development and process-diagnosis logic validation.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.simplefilter(action="ignore", category=DeprecationWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

# ============================================================
# Import V3 simulator core
# ============================================================
try:
    from generate_training_dataset import (
        C0,
        FREQ_MIN_GHZ,
        FREQ_MAX_GHZ,
        KEY_FREQS_GHZ,
        build_default_pns,
        sample_case_parameters,
        simulate_vna,
        simulate_tdr,
        simulate_deltal,
        extract_features_from_curves,
        add_v3_stage_features,
    )
except Exception as exc:  # pragma: no cover - user-friendly direct-run message
    raise ImportError(
        "Cannot import generate_training_dataset.py.\n"
        "Please put generate_validation_lot.py "
        "in the same folder as generate_training_dataset.py.\n"
        f"Original import error: {exc}"
    )


# ============================================================
# USER SETTINGS
# ============================================================

RUN_OUTPUT_DIR = "si_simulated_dataset_v4_pn06_lot_aware"
RUN_SEED = 84

# Keep the default dataset size similar to the previous 6000-case V3 dataset:
# 30 lots x 10 boards x 20 test points = 6000 measurement cases.
RUN_LOT_COUNT = 30
RUN_TOTAL_PANELS_PER_LOT = 60
RUN_SAMPLED_BOARDS_PER_LOT = 10
RUN_TEST_POINTS_PER_BOARD = 20

RUN_FREQ_POINTS = 401
RUN_TDR_POINTS = 301
RUN_FREQ_MAX_GHZ = 40.0
RUN_SAVE_RAW_CURVES = True

# PN06 only for the lot-aware production-batch dataset.
TARGET_PN_ID = "PN06"

# If True, V4 will overwrite existing CSV files in the output folder.
OVERWRITE_OUTPUT = True

# Lot pattern distribution. Values are normalized automatically.
LOT_PATTERN_WEIGHTS = {
    "normal_lot": 0.14,
    "lot_global_loss_roughness": 0.14,
    "lot_global_loss_high_df": 0.14,
    "lot_global_impedance_width_shift": 0.14,
    "single_board_issue": 0.12,
    "regional_issue": 0.12,
    "fixed_test_point_structure_issue": 0.10,
    "random_local_defect": 0.12,
    "mixed_issue": 0.08,
}

# How many points are affected for different patterns.
GLOBAL_AFFECT_PROB = 0.85
SINGLE_BOARD_AFFECT_PROB = 0.75
REGIONAL_AFFECT_PROB = 0.80
FIXED_TEST_POINT_AFFECT_PROB = 0.85
RANDOM_LOCAL_DEFECT_RATE = 0.04  # around 8 points out of 200
MIXED_GLOBAL_AFFECT_PROB = 0.45
MIXED_RANDOM_LOCAL_RATE = 0.04


# ============================================================
# Test-point map
# ============================================================

def build_test_point_map(n_points: int = 20) -> pd.DataFrame:
    """Create a repeatable 20-point board map with x/y and region_id."""
    # 5 columns x 4 rows = 20 points.
    xs = np.linspace(15.0, 135.0, 5)
    ys = np.linspace(15.0, 95.0, 4)
    rows = []
    idx = 0
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            idx += 1
            if idx > n_points:
                break
            if c <= 1 and r <= 1:
                region = "left_top"
            elif c >= 3 and r <= 1:
                region = "right_top"
            elif c <= 1 and r >= 2:
                region = "left_bottom"
            elif c >= 3 and r >= 2:
                region = "right_bottom"
            else:
                region = "center"

            section_type = "line_coupon"
            if idx in {4, 9, 14, 19}:
                section_type = "via_rich_coupon"
            elif idx in {5, 10, 15, 20}:
                section_type = "return_path_sensitive_coupon"

            rows.append({
                "test_point_id": f"TP{idx:02d}",
                "test_point_index": idx,
                "x_mm": round(float(x), 3),
                "y_mm": round(float(y), 3),
                "region_id": region,
                "section_type": section_type,
            })
    return pd.DataFrame(rows)


# ============================================================
# Pattern and parameter helpers
# ============================================================

def weighted_choice(rng: np.random.Generator, weights: Dict[str, float]) -> str:
    keys = list(weights.keys())
    probs = np.array([weights[k] for k in keys], dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(keys, p=probs))


def label_group(label: str) -> str:
    if label in {"golden_normal", "normal_production"}:
        return "normal"
    if label in {"roughness_high", "high_df"}:
        return "loss_related"
    if label == "width_variation":
        return "global_width_shift"
    if label == "local_defect":
        return "local_defect"
    if label == "via_stub":
        return "via_stub"
    if label == "return_path_issue":
        return "return_path_issue"
    return "unknown"


def abnormal_flag(label: str) -> int:
    return 0 if label in {"golden_normal", "normal_production"} else 1


def choose_lot_severity(rng: np.random.Generator) -> str:
    return str(rng.choice(["warning", "mild", "medium", "severe"], p=[0.20, 0.35, 0.30, 0.15]))


def make_lot_context(rng: np.random.Generator, pn, lot_pattern: str) -> Dict:
    """Create lot-level shared parameters for global patterns."""
    severity = choose_lot_severity(rng)
    ctx: Dict[str, object] = {
        "lot_severity": severity,
        "global_loss_type": "none",
        "global_roughness_um": np.nan,
        "global_Df": np.nan,
        "global_width_change_pct": 0.0,
        "fixed_test_points": "",
        "target_board_id": "",
        "target_region_id": "",
    }

    if lot_pattern == "lot_global_loss_roughness":
        ctx["global_loss_type"] = "roughness_high"
        if severity in {"warning", "mild"}:
            ctx["global_roughness_um"] = float(rng.uniform(1.0, 2.0))
        elif severity == "medium":
            ctx["global_roughness_um"] = float(rng.uniform(2.0, 3.5))
        else:
            ctx["global_roughness_um"] = float(rng.uniform(3.5, 5.0))

    elif lot_pattern == "lot_global_loss_high_df":
        ctx["global_loss_type"] = "high_df"
        if severity in {"warning", "mild"}:
            ctx["global_Df"] = float(rng.uniform(0.0050, 0.0060))
        elif severity == "medium":
            ctx["global_Df"] = float(rng.uniform(0.0060, 0.0080))
        else:
            ctx["global_Df"] = float(rng.uniform(0.0080, 0.0120))

    elif lot_pattern == "lot_global_impedance_width_shift":
        direction = rng.choice(["narrow", "wide"])
        if severity == "warning":
            mag = rng.uniform(0.02, 0.05)
        elif severity == "mild":
            mag = rng.uniform(0.05, 0.075)
        elif severity == "medium":
            mag = rng.uniform(0.075, 0.12)
        else:
            mag = rng.uniform(0.12, 0.18)
        ctx["global_width_change_pct"] = float(-100 * mag if direction == "narrow" else 100 * mag)

    return ctx


def apply_lot_overrides(params: Dict, pn, lot_pattern: str, lot_ctx: Dict, rng: np.random.Generator) -> Dict:
    """Make affected global-process measurements internally consistent within a lot."""
    params = params.copy()

    if lot_pattern in {"lot_global_loss_roughness", "mixed_issue"} and lot_ctx.get("global_loss_type") == "roughness_high":
        base = lot_ctx.get("global_roughness_um", np.nan)
        if not pd.isna(base):
            params["roughness_um"] = float(max(0.1, rng.normal(float(base), 0.08)))
            params["severity"] = lot_ctx.get("lot_severity", params.get("severity", "mild"))

    if lot_pattern in {"lot_global_loss_high_df", "mixed_issue"} and lot_ctx.get("global_loss_type") == "high_df":
        base = lot_ctx.get("global_Df", np.nan)
        if not pd.isna(base):
            params["Df"] = float(max(0.0005, rng.normal(float(base), 0.00012)))
            params["severity"] = lot_ctx.get("lot_severity", params.get("severity", "mild"))

    if lot_pattern in {"lot_global_impedance_width_shift", "mixed_issue"} and abs(float(lot_ctx.get("global_width_change_pct", 0.0))) > 1e-9:
        width_change_pct = float(lot_ctx["global_width_change_pct"]) + rng.normal(0.0, 0.35)
        params["width_change_pct"] = float(width_change_pct)
        params["trace_width_um"] = float(pn.trace_width_nominal_um * (1.0 + width_change_pct / 100.0))
        params["severity"] = lot_ctx.get("lot_severity", params.get("severity", "mild"))

    return params


def choose_point_label(
    rng: np.random.Generator,
    lot_pattern: str,
    board_id: str,
    test_point_id: str,
    region_id: str,
    lot_ctx: Dict,
) -> Tuple[str, str, str]:
    """Return (measurement label, point_pattern, issue_scope)."""
    if lot_pattern == "normal_lot":
        return "normal_production", "normal", "normal"

    if lot_pattern == "lot_global_loss_roughness":
        if rng.random() < GLOBAL_AFFECT_PROB:
            return "roughness_high", "loss_related", "lot_global"
        return "normal_production", "normal", "normal"

    if lot_pattern == "lot_global_loss_high_df":
        if rng.random() < GLOBAL_AFFECT_PROB:
            return "high_df", "loss_related", "lot_global"
        return "normal_production", "normal", "normal"

    if lot_pattern == "lot_global_impedance_width_shift":
        if rng.random() < GLOBAL_AFFECT_PROB:
            return "width_variation", "global_width_shift", "lot_global"
        return "normal_production", "normal", "normal"

    if lot_pattern == "single_board_issue":
        if board_id == lot_ctx["target_board_id"] and rng.random() < SINGLE_BOARD_AFFECT_PROB:
            # A single board can show mostly width or loss shift, sometimes local defects.
            label = str(rng.choice(["width_variation", "roughness_high", "high_df", "local_defect"], p=[0.40, 0.20, 0.20, 0.20]))
            return label, label_group(label), "single_board"
        return "normal_production", "normal", "normal"

    if lot_pattern == "regional_issue":
        if region_id == lot_ctx["target_region_id"] and rng.random() < REGIONAL_AFFECT_PROB:
            label = str(rng.choice(["width_variation", "local_defect", "return_path_issue"], p=[0.35, 0.40, 0.25]))
            return label, label_group(label), "regional"
        return "normal_production", "normal", "normal"

    if lot_pattern == "fixed_test_point_structure_issue":
        fixed = set(str(lot_ctx.get("fixed_test_points", "")).split(";"))
        if test_point_id in fixed and rng.random() < FIXED_TEST_POINT_AFFECT_PROB:
            label = str(rng.choice(["via_stub", "return_path_issue"], p=[0.65, 0.35]))
            return label, label_group(label), "fixed_test_point"
        return "normal_production", "normal", "normal"

    if lot_pattern == "random_local_defect":
        if rng.random() < RANDOM_LOCAL_DEFECT_RATE:
            return "local_defect", "local_defect", "random_local"
        return "normal_production", "normal", "normal"

    if lot_pattern == "mixed_issue":
        # A weaker global loss background plus sparse random local issues.
        if rng.random() < MIXED_RANDOM_LOCAL_RATE:
            return "local_defect", "local_defect", "random_local"
        if rng.random() < MIXED_GLOBAL_AFFECT_PROB:
            loss_type = str(lot_ctx.get("global_loss_type", "roughness_high"))
            if loss_type not in {"roughness_high", "high_df"}:
                loss_type = str(rng.choice(["roughness_high", "high_df"]))
            return loss_type, "loss_related", "lot_global"
        return "normal_production", "normal", "normal"

    return "normal_production", "normal", "normal"


# ============================================================
# Aggregation helpers
# ============================================================

def status_to_abnormal(s: pd.Series) -> pd.Series:
    return (~s.fillna("normal").eq("normal")).astype(int)


def create_aggregation_tables(features: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    df = features.copy()
    df["is_measurement_abnormal"] = ((df["label"].map(abnormal_flag) == 1) | status_to_abnormal(df["stage1_overall_status"]).eq(1)).astype(int)
    df["is_loss_related"] = df["label_group"].eq("loss_related").astype(int)
    df["is_impedance_width"] = df["label_group"].eq("global_width_shift").astype(int)
    df["is_local_defect"] = df["label_group"].eq("local_defect").astype(int)
    df["is_via_stub"] = df["label_group"].eq("via_stub").astype(int)
    df["is_return_path"] = df["label_group"].eq("return_path_issue").astype(int)
    df["is_warning_or_worse"] = status_to_abnormal(df["stage1_overall_status"])
    df["is_out_of_spec_or_worse"] = df["stage1_overall_status"].isin(["out_of_spec", "severe"]).astype(int)
    df["is_severe"] = df["stage1_overall_status"].eq("severe").astype(int)

    def rate_table(group_cols: List[str], table_name: str) -> pd.DataFrame:
        g = df.groupby(group_cols, dropna=False)
        out = g.agg(
            total_measurements=("case_id", "count"),
            abnormal_count=("is_measurement_abnormal", "sum"),
            abnormal_rate=("is_measurement_abnormal", "mean"),
            warning_or_worse_rate=("is_warning_or_worse", "mean"),
            out_of_spec_or_worse_rate=("is_out_of_spec_or_worse", "mean"),
            severe_rate=("is_severe", "mean"),
            loss_related_rate=("is_loss_related", "mean"),
            impedance_width_rate=("is_impedance_width", "mean"),
            local_defect_rate=("is_local_defect", "mean"),
            via_stub_rate=("is_via_stub", "mean"),
            return_path_rate=("is_return_path", "mean"),
            avg_Z_mean_target_dev_pct=("Z_mean_target_dev_pct", "mean"),
            avg_Z_max_dev_pct=("Z_max_dev_pct", "mean"),
            avg_loss_excess_ratio=("loss_baseline_excess_ratio_max", "mean"),
            max_loss_excess_ratio=("loss_baseline_excess_ratio_max", "max"),
        ).reset_index()
        out["aggregation_table"] = table_name
        return out

    board_aggr = rate_table(["lot_id", "board_id"], "board_aggregation_features")
    region_aggr = rate_table(["lot_id", "region_id"], "region_aggregation_features")
    tp_aggr = rate_table(["lot_id", "test_point_id"], "test_point_repeat_features")

    lot_base = rate_table(["lot_id"], "lot_aggregation_features")

    # Add concentration and repeat features.
    max_board = board_aggr.groupby("lot_id")["abnormal_rate"].max().rename("max_board_abnormal_rate")
    max_region = region_aggr.groupby("lot_id")["abnormal_rate"].max().rename("max_region_abnormal_rate")
    max_tp = tp_aggr.groupby("lot_id")["abnormal_rate"].max().rename("fixed_test_point_repeat_rate")

    lot_aggr = lot_base.merge(max_board, on="lot_id", how="left")
    lot_aggr = lot_aggr.merge(max_region, on="lot_id", how="left")
    lot_aggr = lot_aggr.merge(max_tp, on="lot_id", how="left")
    lot_aggr["board_concentration_score"] = lot_aggr["max_board_abnormal_rate"] - lot_aggr["abnormal_rate"]
    lot_aggr["regional_concentration_score"] = lot_aggr["max_region_abnormal_rate"] - lot_aggr["abnormal_rate"]
    lot_aggr["randomness_score"] = lot_aggr["local_defect_rate"] * (1.0 - lot_aggr["fixed_test_point_repeat_rate"].fillna(0.0))

    # Attach true lot pattern and a simple rule-based diagnosis for sanity checks.
    lot_meta = df[["lot_id", "true_lot_pattern", "PN_ID", "total_panels", "sampled_boards", "test_points_per_board"]].drop_duplicates("lot_id")
    lot_aggr = lot_aggr.merge(lot_meta, on="lot_id", how="left")
    lot_aggr["rule_based_lot_diagnosis"] = lot_aggr.apply(rule_based_lot_diagnosis, axis=1)

    return {
        "board_aggregation_features": board_aggr,
        "region_aggregation_features": region_aggr,
        "test_point_repeat_features": tp_aggr,
        "lot_aggregation_features": lot_aggr,
    }


def rule_based_lot_diagnosis(row: pd.Series) -> str:
    """Simple interpretable Stage-3 rule baseline."""
    if row["abnormal_rate"] < 0.06:
        return "normal_lot"
    if row["loss_related_rate"] >= 0.55:
        return "lot_global_loss_issue"
    if row["impedance_width_rate"] >= 0.55:
        return "lot_global_impedance_issue"
    if row.get("board_concentration_score", 0.0) >= 0.45 and row.get("max_board_abnormal_rate", 0.0) >= 0.55:
        return "single_board_issue"
    if row.get("regional_concentration_score", 0.0) >= 0.30 and row.get("max_region_abnormal_rate", 0.0) >= 0.35:
        return "regional_issue"
    if row.get("fixed_test_point_repeat_rate", 0.0) >= 0.55 and (row["via_stub_rate"] + row["return_path_rate"]) >= 0.05:
        return "fixed_test_point_structure_issue"
    if row["local_defect_rate"] >= 0.02 and row["abnormal_rate"] < 0.20:
        return "random_local_defect"
    return "mixed_issue"


# ============================================================
# Main generation function
# ============================================================

def generate_lot_aware_dataset(
    output_dir: str | Path = RUN_OUTPUT_DIR,
    seed: int = RUN_SEED,
    lot_count: int = RUN_LOT_COUNT,
    total_panels_per_lot: int = RUN_TOTAL_PANELS_PER_LOT,
    sampled_boards_per_lot: int = RUN_SAMPLED_BOARDS_PER_LOT,
    test_points_per_board: int = RUN_TEST_POINTS_PER_BOARD,
    freq_points: int = RUN_FREQ_POINTS,
    tdr_points: int = RUN_TDR_POINTS,
    freq_max_ghz: float = RUN_FREQ_MAX_GHZ,
    save_raw_curves: bool = RUN_SAVE_RAW_CURVES,
) -> None:
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    aggregation_dir = output_dir / "aggregation"

    if OVERWRITE_OUTPUT:
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        aggregation_dir.mkdir(parents=True, exist_ok=True)
    else:
        raw_dir.mkdir(parents=True, exist_ok=False)
        processed_dir.mkdir(parents=True, exist_ok=False)
        aggregation_dir.mkdir(parents=True, exist_ok=False)

    pns = build_default_pns(6)
    pn_candidates = [p for p in pns if p.PN_ID == TARGET_PN_ID]
    if not pn_candidates:
        raise ValueError(f"TARGET_PN_ID not found in V3 PN configs: {TARGET_PN_ID}")
    pn = pn_candidates[0]

    test_map = build_test_point_map(test_points_per_board)
    freq_ghz = np.linspace(FREQ_MIN_GHZ, freq_max_ghz, freq_points)
    distances_mm = np.linspace(0.0, pn.length_mm, tdr_points)

    lot_rows = []
    board_rows = []
    case_rows = []
    feature_rows = []
    vna_chunks = []
    tdr_chunks = []
    deltal_chunks = []

    case_index = 0

    for lot_idx in range(1, lot_count + 1):
        lot_id = f"{TARGET_PN_ID}_LOT{lot_idx:04d}"
        lot_pattern = weighted_choice(rng, LOT_PATTERN_WEIGHTS)
        lot_ctx = make_lot_context(rng, pn, lot_pattern)

        # Pattern-specific targets.
        board_ids = [f"B{i:02d}" for i in range(1, sampled_boards_per_lot + 1)]
        panel_ids = [f"PNL{int(i):02d}" for i in rng.choice(np.arange(1, total_panels_per_lot + 1), size=sampled_boards_per_lot, replace=False)]

        if lot_pattern == "single_board_issue":
            lot_ctx["target_board_id"] = str(rng.choice(board_ids))
        if lot_pattern == "regional_issue":
            lot_ctx["target_region_id"] = str(rng.choice(sorted(test_map["region_id"].unique())))
        if lot_pattern == "fixed_test_point_structure_issue":
            n_fixed = int(rng.choice([1, 2], p=[0.75, 0.25]))
            fixed_tps = rng.choice(test_map["test_point_id"].to_numpy(), size=n_fixed, replace=False).tolist()
            lot_ctx["fixed_test_points"] = ";".join(fixed_tps)
        if lot_pattern == "mixed_issue":
            # Pick a global-loss background for mixed lots.
            if rng.random() < 0.5:
                lot_ctx["global_loss_type"] = "roughness_high"
                lot_ctx["global_roughness_um"] = float(rng.uniform(1.0, 2.8))
            else:
                lot_ctx["global_loss_type"] = "high_df"
                lot_ctx["global_Df"] = float(rng.uniform(0.0050, 0.0075))

        lot_rows.append({
            "lot_id": lot_id,
            "PN_ID": TARGET_PN_ID,
            "total_panels": total_panels_per_lot,
            "sampled_boards": sampled_boards_per_lot,
            "test_points_per_board": test_points_per_board,
            "total_measurement_points": sampled_boards_per_lot * test_points_per_board,
            "true_lot_pattern": lot_pattern,
            **lot_ctx,
        })

        for board_idx, (board_id, panel_id) in enumerate(zip(board_ids, panel_ids), start=1):
            if lot_pattern == "single_board_issue" and board_id == lot_ctx.get("target_board_id"):
                true_board_pattern = "affected_board"
            else:
                true_board_pattern = "normal_or_background"

            board_rows.append({
                "lot_id": lot_id,
                "PN_ID": TARGET_PN_ID,
                "board_id": board_id,
                "panel_id": panel_id,
                "board_index": board_idx,
                "true_board_pattern": true_board_pattern,
                "true_lot_pattern": lot_pattern,
            })

            for _, tp in test_map.iterrows():
                test_point_id = str(tp["test_point_id"])
                region_id = str(tp["region_id"])
                case_index += 1
                case_id = f"{lot_id}_{board_id}_{test_point_id}_C{case_index:07d}"

                label, point_pattern, issue_scope = choose_point_label(
                    rng=rng,
                    lot_pattern=lot_pattern,
                    board_id=board_id,
                    test_point_id=test_point_id,
                    region_id=region_id,
                    lot_ctx=lot_ctx,
                )

                params = sample_case_parameters(rng, pn, label)
                params["label"] = label
                params = apply_lot_overrides(params, pn, lot_pattern, lot_ctx, rng)

                # Simulate curves for this measurement point.
                vna_df = simulate_vna(rng, pn, params, freq_ghz)
                tdr_df = simulate_tdr(rng, pn, params, distances_mm)
                deltal_df = simulate_deltal(rng, pn, params, freq_ghz)

                meta = {
                    "case_id": case_id,
                    "lot_id": lot_id,
                    "PN_ID": TARGET_PN_ID,
                    "PN_description": pn.description,
                    "line_type": pn.line_type,
                    "board_id": board_id,
                    "panel_id": panel_id,
                    "board_index": board_idx,
                    "test_point_id": test_point_id,
                    "test_point_index": int(tp["test_point_index"]),
                    "x_mm": float(tp["x_mm"]),
                    "y_mm": float(tp["y_mm"]),
                    "region_id": region_id,
                    "section_type": str(tp["section_type"]),
                    "label": label,
                    "label_group": label_group(label),
                    "point_pattern": point_pattern,
                    "issue_scope": issue_scope,
                    "true_lot_pattern": lot_pattern,
                    "true_board_pattern": true_board_pattern,
                    "Z0_target": pn.Z0_target,
                    "length_mm": pn.length_mm,
                    "Dk_nominal": pn.Dk_nominal,
                    "Df_nominal": pn.Df_nominal,
                    "roughness_nominal_um": pn.roughness_nominal_um,
                    "trace_width_nominal_um": pn.trace_width_nominal_um,
                    "copper_thickness_nominal_um": pn.copper_thickness_nominal_um,
                    "short_coupon_mm": pn.short_coupon_mm,
                    "long_coupon_mm": pn.long_coupon_mm,
                    "total_panels": total_panels_per_lot,
                    "sampled_boards": sampled_boards_per_lot,
                    "test_points_per_board": test_points_per_board,
                    **params,
                }
                case_rows.append(meta)

                for df_long in (vna_df, tdr_df, deltal_df):
                    df_long.insert(0, "issue_scope", issue_scope)
                    df_long.insert(0, "test_point_id", test_point_id)
                    df_long.insert(0, "board_id", board_id)
                    df_long.insert(0, "lot_id", lot_id)
                    df_long.insert(0, "label", label)
                    df_long.insert(0, "PN_ID", TARGET_PN_ID)
                    df_long.insert(0, "case_id", case_id)

                if save_raw_curves:
                    vna_chunks.append(vna_df.drop(columns=["Z_est_ohm"], errors="ignore"))
                    tdr_chunks.append(tdr_df)
                    deltal_chunks.append(deltal_df)

                feature = extract_features_from_curves(
                    case_id=case_id,
                    pn_id=TARGET_PN_ID,
                    label=label,
                    vna_df=vna_df,
                    tdr_df=tdr_df,
                    deltal_df=deltal_df,
                    z0_target=pn.Z0_target,
                )
                # Add hierarchy info early; it will survive V3 stage-feature enrichment.
                feature.update({
                    "lot_id": lot_id,
                    "board_id": board_id,
                    "panel_id": panel_id,
                    "board_index": board_idx,
                    "test_point_id": test_point_id,
                    "test_point_index": int(tp["test_point_index"]),
                    "x_mm": float(tp["x_mm"]),
                    "y_mm": float(tp["y_mm"]),
                    "region_id": region_id,
                    "section_type": str(tp["section_type"]),
                    "label_group": label_group(label),
                    "point_pattern": point_pattern,
                    "issue_scope": issue_scope,
                    "true_lot_pattern": lot_pattern,
                    "true_board_pattern": true_board_pattern,
                    "total_panels": total_panels_per_lot,
                    "sampled_boards": sampled_boards_per_lot,
                    "test_points_per_board": test_points_per_board,
                })
                feature_rows.append(feature)

        print(f"[LOT] {lot_idx:04d}/{lot_count:04d} {lot_id} pattern={lot_pattern}")

    # DataFrames.
    lot_metadata = pd.DataFrame(lot_rows)
    board_metadata = pd.DataFrame(board_rows)
    case_metadata = pd.DataFrame(case_rows)
    features = pd.DataFrame(feature_rows)

    # Add V3 Stage 1/2 features. This merges metadata by case_id.
    features = add_v3_stage_features(features, case_metadata, [pn])

    # Preserve hierarchy columns if V3 merge created suffixes.
    for col in [
        "lot_id", "board_id", "panel_id", "board_index", "test_point_id", "test_point_index",
        "x_mm", "y_mm", "region_id", "section_type", "label_group", "point_pattern", "issue_scope",
        "true_lot_pattern", "true_board_pattern", "total_panels", "sampled_boards", "test_points_per_board",
    ]:
        if col not in features.columns and f"{col}_meta" in features.columns:
            features[col] = features[f"{col}_meta"]

    aggregation_tables = create_aggregation_tables(features)

    # Save metadata and map.
    lot_metadata.to_csv(raw_dir / "lot_metadata.csv", index=False, encoding="utf-8-sig")
    board_metadata.to_csv(raw_dir / "board_metadata.csv", index=False, encoding="utf-8-sig")
    test_map.to_csv(raw_dir / "test_point_map.csv", index=False, encoding="utf-8-sig")
    case_metadata.to_csv(raw_dir / "case_metadata.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([asdict(pn)]).to_csv(raw_dir / "pn_config.csv", index=False, encoding="utf-8-sig")

    if save_raw_curves:
        pd.concat(vna_chunks, ignore_index=True).to_csv(raw_dir / "vna_sparameter_long.csv", index=False, encoding="utf-8-sig")
        pd.concat(tdr_chunks, ignore_index=True).to_csv(raw_dir / "tdr_impedance_long.csv", index=False, encoding="utf-8-sig")
        pd.concat(deltal_chunks, ignore_index=True).to_csv(raw_dir / "deltal_loss_long.csv", index=False, encoding="utf-8-sig")

    # Save processed feature tables.
    features.to_csv(processed_dir / "si_fingerprint_features.csv", index=False, encoding="utf-8-sig")

    stage1_cols = [c for c in features.columns if (
        c in {"case_id", "lot_id", "board_id", "test_point_id", "PN_ID", "label", "label_group", "severity", "defect_type",
              "stage1_overall_status", "stage1_impedance_status", "stage1_loss_status", "stage1_overall_risk_score",
              "true_lot_pattern", "issue_scope", "region_id"}
        or c.endswith("_target_dev_pct")
        or c.endswith("_target_dev_ohm")
        or c.endswith("_pn_robust_z")
        or c.endswith("_pn_baseline_dev")
        or "risk" in c
        or "theory_dev" in c
        or "theory_excess" in c
    )]
    stage2_cols = [c for c in features.columns if (
        c in {"case_id", "lot_id", "board_id", "test_point_id", "PN_ID", "label", "label_group", "severity", "defect_type",
              "defect_position_mm", "defect_length_mm", "true_lot_pattern", "issue_scope", "region_id"}
        or c.startswith("TDR_")
        or c.startswith("notch_")
        or c.startswith("S21_ripple")
        or c.startswith("RL_")
        or c.startswith("IL_slope")
        or c.startswith("DeltaL_slope")
    )]

    features[stage1_cols].to_csv(processed_dir / "si_stage1_deviation_features.csv", index=False, encoding="utf-8-sig")
    features[stage2_cols].to_csv(processed_dir / "si_stage2_shape_features.csv", index=False, encoding="utf-8-sig")

    for name, table in aggregation_tables.items():
        table.to_csv(aggregation_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame([
        {"item": "seed", "value": seed},
        {"item": "PN_ID", "value": TARGET_PN_ID},
        {"item": "lot_count", "value": lot_count},
        {"item": "total_panels_per_lot", "value": total_panels_per_lot},
        {"item": "sampled_boards_per_lot", "value": sampled_boards_per_lot},
        {"item": "test_points_per_board", "value": test_points_per_board},
        {"item": "total_measurement_cases", "value": len(case_metadata)},
        {"item": "freq_points", "value": freq_points},
        {"item": "tdr_points", "value": tdr_points},
        {"item": "freq_min_GHz", "value": FREQ_MIN_GHZ},
        {"item": "freq_max_GHz", "value": freq_max_ghz},
        {"item": "save_raw_curves", "value": save_raw_curves},
    ])
    pattern_counts = lot_metadata["true_lot_pattern"].value_counts().rename_axis("pattern").reset_index(name="count")
    summary.to_csv(output_dir / "generation_summary.csv", index=False, encoding="utf-8-sig")
    pattern_counts.to_csv(output_dir / "lot_pattern_counts.csv", index=False, encoding="utf-8-sig")

    print("\nDataset generated successfully.")
    print(f"Output directory          : {output_dir.resolve()}")
    print(f"Lots                      : {len(lot_metadata):,}")
    print(f"Measurement cases         : {len(case_metadata):,}")
    if save_raw_curves:
        print(f"VNA rows                  : {len(vna_chunks) * freq_points:,}")
        print(f"TDR rows                  : {len(tdr_chunks) * tdr_points:,}")
        print(f"Delta-L rows              : {len(deltal_chunks) * freq_points:,}")
    print(f"Feature rows              : {len(features):,}")
    print("Lot pattern counts:")
    print(pattern_counts.to_string(index=False))


def main() -> None:
    print("=== V4 PN06 Lot-Aware SI Simulation Data Generator ===")
    print(f"RUN_OUTPUT_DIR            : {RUN_OUTPUT_DIR}")
    print(f"TARGET_PN_ID              : {TARGET_PN_ID}")
    print(f"RUN_LOT_COUNT             : {RUN_LOT_COUNT}")
    print(f"BOARDS x TEST_POINTS      : {RUN_SAMPLED_BOARDS_PER_LOT} x {RUN_TEST_POINTS_PER_BOARD}")
    print(f"TOTAL MEASUREMENT CASES   : {RUN_LOT_COUNT * RUN_SAMPLED_BOARDS_PER_LOT * RUN_TEST_POINTS_PER_BOARD}")
    print(f"RUN_FREQ_MAX_GHZ          : {RUN_FREQ_MAX_GHZ}")
    print(f"RUN_TDR_POINTS            : {RUN_TDR_POINTS}")
    generate_lot_aware_dataset()


if __name__ == "__main__":
    main()
