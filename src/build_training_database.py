"""
build_training_database.py

Direct-run converter for the V3 two-stage high-speed PCB SI simulated dataset.

How to use in VS Code:
1. Put this file in your project folder.
2. Make sure your V3 dataset folder exists, usually:
       si_simulated_dataset_v3/
3. Press Run.

Expected V3 dataset folder structure:

si_simulated_dataset_v3/
├── generation_summary.csv
├── raw/
│   ├── case_metadata.csv
│   ├── pn_config.csv
│   ├── vna_sparameter_long.csv
│   ├── tdr_impedance_long.csv
│   └── deltal_loss_long.csv
└── processed/
    ├── si_fingerprint_features.csv
    ├── si_stage1_deviation_features.csv
    └── si_stage2_shape_features.csv

Output:
    si_simulated_dataset_v3/si_simulated_dataset_v3.sqlite

V3 additions:
- Imports Stage 1 baseline/deviation feature table.
- Imports Stage 2 shape/local-anomaly feature table.
- Creates views:
    view_case_summary
    view_stage1_deviation_summary
    view_stage2_shape_summary
    view_two_stage_summary
- Compatible with V2-style datasets if the V3 stage tables are missing.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


# ============================================================
# USER SETTINGS
# ============================================================
# Change these values directly in the code if needed.
# Environment variables are optional and only for convenience.

DATASET_DIR = os.getenv("SI_DATASET_DIR", "si_simulated_dataset_v3")
DB_NAME = os.getenv("SI_DB_NAME", "si_simulated_dataset_v3.sqlite")
CHUNKSIZE = 100_000

# If True, the script will first look for DATASET_DIR. If not found, it will
# try common fallback dataset folder names.
AUTO_FIND_DATASET = True
DATASET_FALLBACKS = [
    "si_simulated_dataset_v3",
    "si_v3_test",
    "si_simulated_dataset_v2",
    "si_simulated_dataset",
    "si_test_v2",
]

# If True, delete the old SQLite file before rebuilding it.
OVERWRITE_DB_FILE = True


# ============================================================
# CSV-to-table mapping
# ============================================================

TABLE_SPECS = [
    {"csv": "raw/case_metadata.csv", "table": "case_metadata", "chunksize": None},
    {"csv": "raw/pn_config.csv", "table": "pn_config", "chunksize": None},
    {"csv": "raw/vna_sparameter_long.csv", "table": "vna_sparameter_long", "chunksize": CHUNKSIZE},
    {"csv": "raw/tdr_impedance_long.csv", "table": "tdr_impedance_long", "chunksize": CHUNKSIZE},
    {"csv": "raw/deltal_loss_long.csv", "table": "deltal_loss_long", "chunksize": CHUNKSIZE},
    {"csv": "processed/si_fingerprint_features.csv", "table": "si_fingerprint_features", "chunksize": None},
    {"csv": "processed/si_stage1_deviation_features.csv", "table": "si_stage1_deviation_features", "chunksize": None},
    {"csv": "processed/si_stage2_shape_features.csv", "table": "si_stage2_shape_features", "chunksize": None},
]


# ============================================================
# Helper functions
# ============================================================

def resolve_dataset_dir() -> Path:
    dataset_dir = Path(DATASET_DIR)
    if dataset_dir.exists():
        return dataset_dir

    if AUTO_FIND_DATASET:
        for candidate in DATASET_FALLBACKS:
            path = Path(candidate)
            if path.exists():
                return path

    raise FileNotFoundError(
        "Dataset directory not found.\n"
        f"Current DATASET_DIR = {DATASET_DIR!r}\n"
        "Please change DATASET_DIR in the USER SETTINGS section."
    )


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
        (table,),
    )
    return cur.fetchone() is not None


def view_exists(conn: sqlite3.Connection, view: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='view' AND name=?;",
        (view,),
    )
    return cur.fetchone() is not None


def get_table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    if not table_exists(conn, table):
        return []
    rows = conn.execute(f'PRAGMA table_info({quote_ident(table)});').fetchall()
    return [row[1] for row in rows]


def import_csv_to_table(
    conn: sqlite3.Connection,
    csv_path: Path,
    table_name: str,
    chunksize: int | None,
) -> None:
    if not csv_path.exists():
        print(f"[SKIP] Missing CSV: {csv_path}")
        return

    print(f"[IMPORT] {csv_path} -> {table_name}")
    conn.execute(f'DROP TABLE IF EXISTS {quote_ident(table_name)};')
    conn.commit()

    if chunksize is None:
        df = pd.read_csv(csv_path)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"         rows: {len(df):,}")
        return

    total = 0
    first = True
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        chunk.to_sql(
            table_name,
            conn,
            if_exists="replace" if first else "append",
            index=False,
        )
        total += len(chunk)
        first = False
        print(f"         imported rows: {total:,}", end="\r")
    print(f"         rows: {total:,}")


def optimize_sqlite(conn: sqlite3.Connection) -> None:
    # Good pragmas for a local analysis database.
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.execute("PRAGMA cache_size = -200000;")  # about 200 MB cache
    conn.commit()


def create_index_if_columns_exist(
    conn: sqlite3.Connection,
    table: str,
    index_name: str,
    columns: Sequence[str],
) -> None:
    if not table_exists(conn, table):
        return
    existing = set(get_table_columns(conn, table))
    if not all(col in existing for col in columns):
        return
    col_sql = ", ".join(quote_ident(col) for col in columns)
    sql = f"CREATE INDEX IF NOT EXISTS {quote_ident(index_name)} ON {quote_ident(table)}({col_sql});"
    conn.execute(sql)


def create_indexes(conn: sqlite3.Connection) -> None:
    print("[INDEX] Creating indexes...")

    index_specs = [
        # Metadata
        ("case_metadata", "idx_case_metadata_case_id", ["case_id"]),
        ("case_metadata", "idx_case_metadata_pn_label", ["PN_ID", "label"]),
        ("case_metadata", "idx_case_metadata_label", ["label"]),

        # PN config
        ("pn_config", "idx_pn_config_pn_id", ["PN_ID"]),

        # VNA
        ("vna_sparameter_long", "idx_vna_case_id", ["case_id"]),
        ("vna_sparameter_long", "idx_vna_pn_label", ["PN_ID", "label"]),
        ("vna_sparameter_long", "idx_vna_freq", ["frequency_GHz"]),
        ("vna_sparameter_long", "idx_vna_case_freq", ["case_id", "frequency_GHz"]),

        # TDR
        ("tdr_impedance_long", "idx_tdr_case_id", ["case_id"]),
        ("tdr_impedance_long", "idx_tdr_pn_label", ["PN_ID", "label"]),
        ("tdr_impedance_long", "idx_tdr_distance", ["distance_mm"]),
        ("tdr_impedance_long", "idx_tdr_case_distance", ["case_id", "distance_mm"]),

        # Delta-L
        ("deltal_loss_long", "idx_deltal_case_id", ["case_id"]),
        ("deltal_loss_long", "idx_deltal_pn_label", ["PN_ID", "label"]),
        ("deltal_loss_long", "idx_deltal_freq", ["frequency_GHz"]),
        ("deltal_loss_long", "idx_deltal_case_freq", ["case_id", "frequency_GHz"]),

        # V2/V3 all-in feature table
        ("si_fingerprint_features", "idx_features_case_id", ["case_id"]),
        ("si_fingerprint_features", "idx_features_pn_label", ["PN_ID", "label"]),
        ("si_fingerprint_features", "idx_features_label", ["label"]),
        ("si_fingerprint_features", "idx_features_stage1_status", ["stage1_overall_status"]),

        # V3 Stage 1
        ("si_stage1_deviation_features", "idx_stage1_case_id", ["case_id"]),
        ("si_stage1_deviation_features", "idx_stage1_pn_label", ["PN_ID", "label"]),
        ("si_stage1_deviation_features", "idx_stage1_label", ["label"]),
        ("si_stage1_deviation_features", "idx_stage1_overall_status", ["stage1_overall_status"]),
        ("si_stage1_deviation_features", "idx_stage1_impedance_status", ["stage1_impedance_status"]),
        ("si_stage1_deviation_features", "idx_stage1_loss_status", ["stage1_loss_status"]),

        # V3 Stage 2
        ("si_stage2_shape_features", "idx_stage2_case_id", ["case_id"]),
        ("si_stage2_shape_features", "idx_stage2_pn_label", ["PN_ID", "label"]),
        ("si_stage2_shape_features", "idx_stage2_label", ["label"]),
        ("si_stage2_shape_features", "idx_stage2_defect_type", ["defect_type"]),
        ("si_stage2_shape_features", "idx_stage2_severity", ["severity"]),
    ]

    for table, index_name, columns in index_specs:
        try:
            create_index_if_columns_exist(conn, table, index_name, columns)
        except sqlite3.OperationalError as exc:
            print(f"[WARN] Index creation skipped for {index_name}: {exc}")

    conn.commit()


def add_existing_columns(
    select_parts: list[str],
    available_columns: Iterable[str],
    table_alias: str,
    desired_columns: Iterable[str],
    output_prefix: str = "",
    used_aliases: set[str] | None = None,
) -> None:
    available = set(available_columns)
    if used_aliases is None:
        used_aliases = set()
    for col in desired_columns:
        if col not in available:
            continue
        alias = f"{output_prefix}{col}"
        if alias in used_aliases:
            continue
        used_aliases.add(alias)
        select_parts.append(f'{table_alias}.{quote_ident(col)} AS {quote_ident(alias)}')


def add_all_columns(
    select_parts: list[str],
    available_columns: Iterable[str],
    table_alias: str,
    output_prefix: str = "",
    skip_columns: Iterable[str] | None = None,
    used_aliases: set[str] | None = None,
) -> None:
    if used_aliases is None:
        used_aliases = set()
    skip = set(skip_columns or [])
    for col in available_columns:
        if col in skip:
            continue
        alias = f"{output_prefix}{col}"
        if alias in used_aliases:
            continue
        used_aliases.add(alias)
        select_parts.append(f'{table_alias}.{quote_ident(col)} AS {quote_ident(alias)}')


def create_case_summary_view(conn: sqlite3.Connection) -> None:
    if not table_exists(conn, "si_fingerprint_features"):
        print("[VIEW] Feature table not found; view_case_summary skipped.")
        return

    print("[VIEW] Creating view_case_summary...")
    feature_cols = get_table_columns(conn, "si_fingerprint_features")
    metadata_cols = get_table_columns(conn, "case_metadata")

    select_parts: list[str] = []
    used_aliases: set[str] = set()

    # Keep all V3 fingerprint columns. This table already includes most Stage 1/2 columns.
    add_all_columns(select_parts, feature_cols, "f", used_aliases=used_aliases)

    # Add metadata columns that are not already present. Prefix only if duplicated.
    if table_exists(conn, "case_metadata"):
        for col in metadata_cols:
            if col in {"case_id", "PN_ID", "label"}:
                continue
            prefix = "meta_" if col in used_aliases else ""
            add_existing_columns(select_parts, metadata_cols, "m", [col], output_prefix=prefix, used_aliases=used_aliases)

    if not select_parts:
        print("[VIEW] No usable columns found; view_case_summary skipped.")
        return

    join_sql = ""
    if table_exists(conn, "case_metadata"):
        join_sql = "\nLEFT JOIN case_metadata m ON f.case_id = m.case_id"

    select_sql = ",\n        ".join(select_parts)
    conn.execute("DROP VIEW IF EXISTS view_case_summary;")
    conn.execute(
        f"""
CREATE VIEW view_case_summary AS
SELECT
        {select_sql}
FROM si_fingerprint_features f{join_sql};
"""
    )
    conn.commit()
    n_rows = conn.execute("SELECT COUNT(*) FROM view_case_summary;").fetchone()[0]
    print(f"[VIEW] view_case_summary rows: {n_rows:,}")


def create_stage_view(
    conn: sqlite3.Connection,
    source_table: str,
    view_name: str,
    label: str,
) -> None:
    if not table_exists(conn, source_table):
        print(f"[VIEW] {source_table} not found; {view_name} skipped.")
        return

    print(f"[VIEW] Creating {view_name}...")
    source_cols = get_table_columns(conn, source_table)
    metadata_cols = get_table_columns(conn, "case_metadata")

    select_parts: list[str] = []
    used_aliases: set[str] = set()
    add_all_columns(select_parts, source_cols, "s", used_aliases=used_aliases)

    if table_exists(conn, "case_metadata"):
        # Add useful physical metadata, but avoid collisions.
        metadata_desired_cols = [
            "PN_description",
            "line_type",
            "Z0_target",
            "length_mm",
            "Dk_nominal",
            "Df_nominal",
            "roughness_nominal_um",
            "trace_width_nominal_um",
            "copper_thickness_nominal_um",
            "short_coupon_mm",
            "long_coupon_mm",
            "Dk",
            "Df",
            "roughness_um",
            "trace_width_um",
            "copper_thickness_um",
            "dielectric_thickness_factor",
            "width_change_pct",
            "defect_position_mm",
            "defect_length_mm",
            "stub_length_mm",
            "return_path_extra_L_pH",
        ]
        for col in metadata_desired_cols:
            prefix = "meta_" if col in used_aliases else ""
            add_existing_columns(select_parts, metadata_cols, "m", [col], output_prefix=prefix, used_aliases=used_aliases)

    join_sql = ""
    if table_exists(conn, "case_metadata"):
        join_sql = "\nLEFT JOIN case_metadata m ON s.case_id = m.case_id"

    select_sql = ",\n        ".join(select_parts)
    conn.execute(f"DROP VIEW IF EXISTS {quote_ident(view_name)};")
    conn.execute(
        f"""
CREATE VIEW {quote_ident(view_name)} AS
SELECT
        {select_sql}
FROM {quote_ident(source_table)} s{join_sql};
"""
    )
    conn.commit()
    n_rows = conn.execute(f"SELECT COUNT(*) FROM {quote_ident(view_name)};").fetchone()[0]
    print(f"[VIEW] {view_name} rows: {n_rows:,}")


def create_two_stage_summary_view(conn: sqlite3.Connection) -> None:
    if not table_exists(conn, "si_fingerprint_features"):
        print("[VIEW] Feature table not found; view_two_stage_summary skipped.")
        return

    print("[VIEW] Creating view_two_stage_summary...")

    f_cols = get_table_columns(conn, "si_fingerprint_features")
    s1_cols = get_table_columns(conn, "si_stage1_deviation_features")
    s2_cols = get_table_columns(conn, "si_stage2_shape_features")
    m_cols = get_table_columns(conn, "case_metadata")

    select_parts: list[str] = []
    used_aliases: set[str] = set()

    # Identity columns.
    add_existing_columns(select_parts, f_cols, "f", ["case_id", "PN_ID", "label"], used_aliases=used_aliases)

    # Basic metadata / ground-truth physical parameters.
    metadata_cols = [
        "severity",
        "defect_type",
        "Z0_target",
        "length_mm",
        "Dk",
        "Df",
        "roughness_um",
        "trace_width_um",
        "width_change_pct",
        "defect_position_mm",
        "defect_length_mm",
        "stub_length_mm",
        "return_path_extra_L_pH",
    ]
    # Prefer metadata table for physical parameters; fall back to fingerprint if missing.
    if table_exists(conn, "case_metadata"):
        add_existing_columns(select_parts, m_cols, "m", metadata_cols, used_aliases=used_aliases)
    add_existing_columns(select_parts, f_cols, "f", metadata_cols, used_aliases=used_aliases)

    # Core loss/impedance features.
    core_fingerprint_cols = [
        "IL_16GHz",
        "IL_20GHz",
        "IL_28GHz",
        "IL_32GHz",
        "IL_40GHz",
        "DeltaL_16GHz_dB_per_in",
        "DeltaL_20GHz_dB_per_in",
        "DeltaL_28GHz_dB_per_in",
        "DeltaL_32GHz_dB_per_in",
        "DeltaL_40GHz_dB_per_in",
        "IL_slope_16GHz_to_20GHz",
        "IL_slope_20GHz_to_28GHz",
        "IL_slope_28GHz_to_40GHz",
        "DeltaL_slope_16GHz_to_20GHz",
        "DeltaL_slope_20GHz_to_28GHz",
        "DeltaL_slope_28GHz_to_40GHz",
        "Z_mean",
        "Z_max_dev",
        "TDR_peak",
        "TDR_dip",
        "TDR_energy",
        "RL_min",
        "RL_mean",
        "RL_ripple",
        "notch_depth",
        "notch_frequency",
        "S21_ripple_0p1GHz_to_20GHz",
        "S21_ripple_20GHz_to_40GHz",
    ]
    add_existing_columns(select_parts, f_cols, "f", core_fingerprint_cols, used_aliases=used_aliases)

    # Stage 1 deviation/risk columns.
    stage1_cols = [
        "Z_mean_target_dev_ohm",
        "Z_mean_target_dev_pct",
        "impedance_mean_risk_score",
        "impedance_local_risk_score",
        "impedance_mean_risk",
        "impedance_local_risk",
        "loss_baseline_risk_score",
        "loss_baseline_risk",
        "loss_baseline_excess_ratio_max",
        "stage1_impedance_status",
        "stage1_loss_status",
        "stage1_overall_risk_score",
        "stage1_overall_status",
        "IL_28GHz_theory_dev_dB",
        "IL_40GHz_theory_dev_dB",
        "DeltaL_28GHz_dB_per_in_theory_excess_ratio",
        "DeltaL_40GHz_dB_per_in_theory_excess_ratio",
        "IL_28GHz_pn_baseline_dev",
        "IL_28GHz_pn_robust_z",
        "IL_40GHz_pn_baseline_dev",
        "IL_40GHz_pn_robust_z",
        "DeltaL_28GHz_dB_per_in_pn_baseline_dev",
        "DeltaL_28GHz_dB_per_in_pn_robust_z",
        "DeltaL_40GHz_dB_per_in_pn_baseline_dev",
        "DeltaL_40GHz_dB_per_in_pn_robust_z",
        "Z_mean_pn_baseline_dev",
        "Z_mean_pn_robust_z",
        "Z_max_dev_pn_baseline_dev",
        "Z_max_dev_pn_robust_z",
        "TDR_peak_abs_ohm_pn_baseline_dev",
        "TDR_peak_abs_ohm_pn_robust_z",
    ]
    if table_exists(conn, "si_stage1_deviation_features"):
        add_existing_columns(select_parts, s1_cols, "s1", stage1_cols, used_aliases=used_aliases)
    add_existing_columns(select_parts, f_cols, "f", stage1_cols, used_aliases=used_aliases)

    # Stage 2 shape columns.
    stage2_cols = [
        "TDR_peak_to_peak",
        "TDR_peak_abs_ohm",
        "TDR_peak_polarity",
        "TDR_local_peak_width_mm",
        "TDR_local_peak_area_ohm_mm",
        "TDR_num_regions_above_3pct",
        "TDR_num_regions_above_7pct",
        "TDR_warning_threshold_ohm",
        "TDR_spec_threshold_ohm",
        "TDR_peak_pct",
        "TDR_dip_pct",
        "TDR_peak_abs_pct",
        "notch_depth_0p1GHz_to_20GHz",
        "notch_frequency_0p1GHz_to_20GHz",
        "notch_depth_20GHz_to_40GHz",
        "notch_frequency_20GHz_to_40GHz",
        "S21_ripple_0p1GHz_to_20GHz",
        "S21_ripple_20GHz_to_40GHz",
        "RL_min_pn_robust_z",
        "RL_ripple_pn_robust_z",
        "notch_depth_pn_robust_z",
        "S21_ripple_20GHz_to_40GHz_pn_robust_z",
    ]
    if table_exists(conn, "si_stage2_shape_features"):
        add_existing_columns(select_parts, s2_cols, "s2", stage2_cols, used_aliases=used_aliases)
    add_existing_columns(select_parts, f_cols, "f", stage2_cols, used_aliases=used_aliases)

    join_sql = ""
    if table_exists(conn, "case_metadata"):
        join_sql += "\nLEFT JOIN case_metadata m ON f.case_id = m.case_id"
    if table_exists(conn, "si_stage1_deviation_features"):
        join_sql += "\nLEFT JOIN si_stage1_deviation_features s1 ON f.case_id = s1.case_id"
    if table_exists(conn, "si_stage2_shape_features"):
        join_sql += "\nLEFT JOIN si_stage2_shape_features s2 ON f.case_id = s2.case_id"

    if not select_parts:
        print("[VIEW] No usable columns found; view_two_stage_summary skipped.")
        return

    select_sql = ",\n        ".join(select_parts)
    conn.execute("DROP VIEW IF EXISTS view_two_stage_summary;")
    conn.execute(
        f"""
CREATE VIEW view_two_stage_summary AS
SELECT
        {select_sql}
FROM si_fingerprint_features f{join_sql};
"""
    )
    conn.commit()
    n_rows = conn.execute("SELECT COUNT(*) FROM view_two_stage_summary;").fetchone()[0]
    print(f"[VIEW] view_two_stage_summary rows: {n_rows:,}")


def create_views(conn: sqlite3.Connection) -> None:
    # Drop dependent views first for clean repeated runs.
    for view in [
        "view_two_stage_summary",
        "view_stage2_shape_summary",
        "view_stage1_deviation_summary",
        "view_case_summary",
    ]:
        conn.execute(f"DROP VIEW IF EXISTS {quote_ident(view)};")
    conn.commit()

    create_case_summary_view(conn)
    create_stage_view(
        conn,
        source_table="si_stage1_deviation_features",
        view_name="view_stage1_deviation_summary",
        label="Stage 1",
    )
    create_stage_view(
        conn,
        source_table="si_stage2_shape_features",
        view_name="view_stage2_shape_summary",
        label="Stage 2",
    )
    create_two_stage_summary_view(conn)


def create_indexes_and_views(conn: sqlite3.Connection) -> None:
    create_indexes(conn)
    create_views(conn)

    print("[ANALYZE] Optimizing query planner statistics...")
    conn.execute("ANALYZE;")
    conn.commit()


def print_frequency_range(conn: sqlite3.Connection, table: str, label: str) -> None:
    if not table_exists(conn, table):
        return
    cols = get_table_columns(conn, table)
    if "frequency_GHz" not in cols:
        return
    row = conn.execute(
        f"SELECT MIN(frequency_GHz), MAX(frequency_GHz), COUNT(*) FROM {quote_ident(table)};"
    ).fetchone()
    if row is None:
        return
    f_min, f_max, n = row
    print(f"{label:30s}: {f_min:.3f} to {f_max:.3f} GHz, rows={n:,}")


def print_table_row_count(conn: sqlite3.Connection, table: str) -> None:
    if table_exists(conn, table):
        count = conn.execute(f"SELECT COUNT(*) FROM {quote_ident(table)};").fetchone()[0]
        print(f"{table:34s}: {count:,} rows")
    else:
        print(f"{table:34s}: missing")


def print_view_row_count(conn: sqlite3.Connection, view: str) -> None:
    if view_exists(conn, view):
        count = conn.execute(f"SELECT COUNT(*) FROM {quote_ident(view)};").fetchone()[0]
        print(f"{view:34s}: {count:,} rows")
    else:
        print(f"{view:34s}: missing")


def print_feature_column_checks(conn: sqlite3.Connection) -> None:
    if not table_exists(conn, "si_fingerprint_features"):
        return

    cols = get_table_columns(conn, "si_fingerprint_features")
    stage1_cols = get_table_columns(conn, "si_stage1_deviation_features")
    stage2_cols = get_table_columns(conn, "si_stage2_shape_features")

    il_cols = [c for c in cols if c.startswith("IL_")]
    dl_cols = [c for c in cols if c.startswith("DeltaL_")]
    slope_cols = [c for c in cols if "slope" in c]
    notch_ripple_cols = [c for c in cols if c.startswith("notch_") or "ripple" in c]
    stage1_risk_cols = [c for c in stage1_cols if "risk" in c or "status" in c]
    stage2_shape_cols = [
        c for c in stage2_cols
        if c.startswith("TDR_") or c.startswith("RL_") or c.startswith("notch_") or "ripple" in c
    ]

    print("\n=== V3 feature checks ===")
    print(f"Total fingerprint columns    : {len(cols):,}")
    print(f"IL feature columns           : {len(il_cols):,}")
    print(f"Delta-L feature columns      : {len(dl_cols):,}")
    print(f"Slope feature columns        : {len(slope_cols):,}")
    print(f"Notch/ripple feature columns : {len(notch_ripple_cols):,}")
    print(f"Stage 1 columns              : {len(stage1_cols):,}")
    print(f"Stage 1 risk/status columns  : {len(stage1_risk_cols):,}")
    print(f"Stage 2 columns              : {len(stage2_cols):,}")
    print(f"Stage 2 shape columns        : {len(stage2_shape_cols):,}")

    important_checks = {
        "Fingerprint / frequency-aware": [
            "IL_16GHz",
            "IL_20GHz",
            "IL_28GHz",
            "IL_32GHz",
            "IL_40GHz",
            "DeltaL_20GHz_dB_per_in",
            "DeltaL_28GHz_dB_per_in",
            "DeltaL_40GHz_dB_per_in",
            "IL_slope_20GHz_to_28GHz",
            "IL_slope_28GHz_to_40GHz",
            "S21_ripple_20GHz_to_40GHz",
        ],
        "Stage 1 / baseline deviation": [
            "Z_mean_target_dev_pct",
            "impedance_mean_risk_score",
            "impedance_local_risk_score",
            "loss_baseline_risk_score",
            "stage1_impedance_status",
            "stage1_loss_status",
            "stage1_overall_status",
            "IL_28GHz_pn_robust_z",
            "DeltaL_28GHz_dB_per_in_pn_robust_z",
        ],
        "Stage 2 / shape anomaly": [
            "TDR_peak_abs_ohm",
            "TDR_peak_polarity",
            "TDR_local_peak_width_mm",
            "TDR_local_peak_area_ohm_mm",
            "TDR_num_regions_above_3pct",
            "TDR_num_regions_above_7pct",
            "notch_depth_20GHz_to_40GHz",
            "S21_ripple_20GHz_to_40GHz",
        ],
    }

    print("\nImportant V3 columns:")
    for title, required in important_checks.items():
        if title.startswith("Stage 1"):
            available = set(stage1_cols) | set(cols)
        elif title.startswith("Stage 2"):
            available = set(stage2_cols) | set(cols)
        else:
            available = set(cols)
        print(f"\n{title}")
        for col in required:
            mark = "OK" if col in available else "MISSING"
            print(f"  {mark:7s} {col}")


def print_stage_status_counts(conn: sqlite3.Connection) -> None:
    if not table_exists(conn, "si_stage1_deviation_features"):
        return

    print("\n=== Stage 1 status counts ===")
    cols = get_table_columns(conn, "si_stage1_deviation_features")
    for col in ["stage1_impedance_status", "stage1_loss_status", "stage1_overall_status"]:
        if col not in cols:
            continue
        print(f"\n{col}:")
        rows = conn.execute(
            f"""
            SELECT {quote_ident(col)} AS status, COUNT(*) AS n
            FROM si_stage1_deviation_features
            GROUP BY {quote_ident(col)}
            ORDER BY n DESC;
            """
        ).fetchall()
        for status, n in rows:
            print(f"  {str(status):16s}: {n:,}")


def print_basic_checks(conn: sqlite3.Connection) -> None:
    print("\n=== Basic checks ===")
    tables = [
        "case_metadata",
        "pn_config",
        "vna_sparameter_long",
        "tdr_impedance_long",
        "deltal_loss_long",
        "si_fingerprint_features",
        "si_stage1_deviation_features",
        "si_stage2_shape_features",
    ]
    for table in tables:
        print_table_row_count(conn, table)

    print("\n=== Views ===")
    for view in [
        "view_case_summary",
        "view_stage1_deviation_summary",
        "view_stage2_shape_summary",
        "view_two_stage_summary",
    ]:
        print_view_row_count(conn, view)

    print("\n=== Frequency ranges ===")
    print_frequency_range(conn, "vna_sparameter_long", "VNA/S-parameter")
    print_frequency_range(conn, "deltal_loss_long", "Delta-L")

    if table_exists(conn, "case_metadata"):
        print("\nClass counts:")
        rows = conn.execute(
            """
            SELECT label, COUNT(*) AS n
            FROM case_metadata
            GROUP BY label
            ORDER BY n DESC;
            """
        ).fetchall()
        for label, n in rows:
            print(f"{str(label):24s}: {n:,}")

        print("\nPN counts:")
        rows = conn.execute(
            """
            SELECT PN_ID, COUNT(*) AS n
            FROM case_metadata
            GROUP BY PN_ID
            ORDER BY PN_ID;
            """
        ).fetchall()
        for pn, n in rows:
            print(f"{str(pn):8s}: {n:,}")

    print_feature_column_checks(conn)
    print_stage_status_counts(conn)


def main() -> None:
    dataset_dir = resolve_dataset_dir()
    db_path = dataset_dir / DB_NAME

    print("=== SI CSV to SQLite Converter: V3 Two-Stage Direct Run ===")
    print(f"Dataset directory: {dataset_dir.resolve()}")
    print(f"SQLite DB path   : {db_path.resolve()}")

    if OVERWRITE_DB_FILE and db_path.exists():
        print(f"[RESET] Removing old DB: {db_path}")
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    try:
        optimize_sqlite(conn)

        # Drop dependent views first for clean repeated runs.
        for view in [
            "view_two_stage_summary",
            "view_stage2_shape_summary",
            "view_stage1_deviation_summary",
            "view_case_summary",
        ]:
            conn.execute(f"DROP VIEW IF EXISTS {quote_ident(view)};")
        conn.commit()

        for spec in TABLE_SPECS:
            csv_path = dataset_dir / spec["csv"]
            import_csv_to_table(conn, csv_path, spec["table"], spec["chunksize"])

        create_indexes_and_views(conn)
        print_basic_checks(conn)

    finally:
        conn.close()

    print("\nDone.")
    print(f"SQLite database created: {db_path.resolve()}")


if __name__ == "__main__":
    main()
