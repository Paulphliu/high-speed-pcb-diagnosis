"""
build_validation_database.py

Direct-run CSV-to-SQLite converter for the V4 PN06 lot-aware SI dataset.

Expected dataset folder:
si_simulated_dataset_v4_pn06_lot_aware/
├── generation_summary.csv
├── lot_pattern_counts.csv
├── raw/
│   ├── lot_metadata.csv
│   ├── board_metadata.csv
│   ├── test_point_map.csv
│   ├── case_metadata.csv
│   ├── pn_config.csv
│   ├── vna_sparameter_long.csv              optional if raw curves saved
│   ├── tdr_impedance_long.csv               optional if raw curves saved
│   └── deltal_loss_long.csv                 optional if raw curves saved
├── processed/
│   ├── si_fingerprint_features.csv
│   ├── si_stage1_deviation_features.csv
│   └── si_stage2_shape_features.csv
└── aggregation/
    ├── board_aggregation_features.csv
    ├── region_aggregation_features.csv
    ├── test_point_repeat_features.csv
    └── lot_aggregation_features.csv

Output:
    si_simulated_dataset_v4_pn06_lot_aware/si_simulated_dataset_v4_pn06_lot_aware.sqlite
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd


# ============================================================
# USER SETTINGS
# ============================================================

DATASET_DIR = "si_simulated_dataset_v4_pn06_lot_aware"
DB_NAME = "si_simulated_dataset_v4_pn06_lot_aware.sqlite"
CHUNKSIZE = 100_000
OVERWRITE_DB_FILE = True

AUTO_FIND_DATASET = True
DATASET_FALLBACKS = [
    "si_simulated_dataset_v4_pn06_lot_aware",
    "si_v4_test",
    "si_simulated_dataset_v3",
]


# ============================================================
# Table specs
# ============================================================

TABLE_SPECS = [
    {"csv": "generation_summary.csv", "table": "generation_summary", "chunksize": None},
    {"csv": "lot_pattern_counts.csv", "table": "lot_pattern_counts", "chunksize": None},

    {"csv": "raw/lot_metadata.csv", "table": "lot_metadata", "chunksize": None},
    {"csv": "raw/board_metadata.csv", "table": "board_metadata", "chunksize": None},
    {"csv": "raw/test_point_map.csv", "table": "test_point_map", "chunksize": None},
    {"csv": "raw/case_metadata.csv", "table": "case_metadata", "chunksize": None},
    {"csv": "raw/pn_config.csv", "table": "pn_config", "chunksize": None},

    {"csv": "raw/vna_sparameter_long.csv", "table": "vna_sparameter_long", "chunksize": CHUNKSIZE},
    {"csv": "raw/tdr_impedance_long.csv", "table": "tdr_impedance_long", "chunksize": CHUNKSIZE},
    {"csv": "raw/deltal_loss_long.csv", "table": "deltal_loss_long", "chunksize": CHUNKSIZE},

    {"csv": "processed/si_fingerprint_features.csv", "table": "si_fingerprint_features", "chunksize": None},
    {"csv": "processed/si_stage1_deviation_features.csv", "table": "si_stage1_deviation_features", "chunksize": None},
    {"csv": "processed/si_stage2_shape_features.csv", "table": "si_stage2_shape_features", "chunksize": None},

    {"csv": "aggregation/board_aggregation_features.csv", "table": "board_aggregation_features", "chunksize": None},
    {"csv": "aggregation/region_aggregation_features.csv", "table": "region_aggregation_features", "chunksize": None},
    {"csv": "aggregation/test_point_repeat_features.csv", "table": "test_point_repeat_features", "chunksize": None},
    {"csv": "aggregation/lot_aggregation_features.csv", "table": "lot_aggregation_features", "chunksize": None},
]


INDEX_SQL = [
    # Lot hierarchy
    "CREATE INDEX IF NOT EXISTS idx_lot_metadata_lot ON lot_metadata(lot_id);",
    "CREATE INDEX IF NOT EXISTS idx_lot_metadata_pattern ON lot_metadata(true_lot_pattern);",
    "CREATE INDEX IF NOT EXISTS idx_board_metadata_lot_board ON board_metadata(lot_id, board_id);",
    "CREATE INDEX IF NOT EXISTS idx_test_point_map_tp ON test_point_map(test_point_id);",

    # Case metadata
    "CREATE INDEX IF NOT EXISTS idx_case_metadata_case ON case_metadata(case_id);",
    "CREATE INDEX IF NOT EXISTS idx_case_metadata_lot ON case_metadata(lot_id);",
    "CREATE INDEX IF NOT EXISTS idx_case_metadata_lot_board ON case_metadata(lot_id, board_id);",
    "CREATE INDEX IF NOT EXISTS idx_case_metadata_lot_tp ON case_metadata(lot_id, test_point_id);",
    "CREATE INDEX IF NOT EXISTS idx_case_metadata_region ON case_metadata(lot_id, region_id);",
    "CREATE INDEX IF NOT EXISTS idx_case_metadata_label ON case_metadata(label);",
    "CREATE INDEX IF NOT EXISTS idx_case_metadata_issue_scope ON case_metadata(issue_scope);",

    # Raw curves
    "CREATE INDEX IF NOT EXISTS idx_vna_case ON vna_sparameter_long(case_id);",
    "CREATE INDEX IF NOT EXISTS idx_vna_lot ON vna_sparameter_long(lot_id);",
    "CREATE INDEX IF NOT EXISTS idx_vna_case_freq ON vna_sparameter_long(case_id, frequency_GHz);",
    "CREATE INDEX IF NOT EXISTS idx_tdr_case ON tdr_impedance_long(case_id);",
    "CREATE INDEX IF NOT EXISTS idx_tdr_lot ON tdr_impedance_long(lot_id);",
    "CREATE INDEX IF NOT EXISTS idx_tdr_case_dist ON tdr_impedance_long(case_id, distance_mm);",
    "CREATE INDEX IF NOT EXISTS idx_deltal_case ON deltal_loss_long(case_id);",
    "CREATE INDEX IF NOT EXISTS idx_deltal_lot ON deltal_loss_long(lot_id);",
    "CREATE INDEX IF NOT EXISTS idx_deltal_case_freq ON deltal_loss_long(case_id, frequency_GHz);",

    # Feature tables
    "CREATE INDEX IF NOT EXISTS idx_features_case ON si_fingerprint_features(case_id);",
    "CREATE INDEX IF NOT EXISTS idx_features_lot ON si_fingerprint_features(lot_id);",
    "CREATE INDEX IF NOT EXISTS idx_features_lot_board ON si_fingerprint_features(lot_id, board_id);",
    "CREATE INDEX IF NOT EXISTS idx_features_lot_tp ON si_fingerprint_features(lot_id, test_point_id);",
    "CREATE INDEX IF NOT EXISTS idx_features_label ON si_fingerprint_features(label);",
    "CREATE INDEX IF NOT EXISTS idx_stage1_case ON si_stage1_deviation_features(case_id);",
    "CREATE INDEX IF NOT EXISTS idx_stage1_lot ON si_stage1_deviation_features(lot_id);",
    "CREATE INDEX IF NOT EXISTS idx_stage2_case ON si_stage2_shape_features(case_id);",
    "CREATE INDEX IF NOT EXISTS idx_stage2_lot ON si_stage2_shape_features(lot_id);",

    # Aggregation tables
    "CREATE INDEX IF NOT EXISTS idx_lot_aggr_lot ON lot_aggregation_features(lot_id);",
    "CREATE INDEX IF NOT EXISTS idx_lot_aggr_true ON lot_aggregation_features(true_lot_pattern);",
    "CREATE INDEX IF NOT EXISTS idx_lot_aggr_rule ON lot_aggregation_features(rule_based_lot_diagnosis);",
    "CREATE INDEX IF NOT EXISTS idx_board_aggr_lot_board ON board_aggregation_features(lot_id, board_id);",
    "CREATE INDEX IF NOT EXISTS idx_region_aggr_lot_region ON region_aggregation_features(lot_id, region_id);",
    "CREATE INDEX IF NOT EXISTS idx_tp_aggr_lot_tp ON test_point_repeat_features(lot_id, test_point_id);",
]


# ============================================================
# Helpers
# ============================================================

def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def resolve_dataset_dir() -> Path:
    dataset_dir = Path(DATASET_DIR)
    if dataset_dir.exists():
        return dataset_dir
    if AUTO_FIND_DATASET:
        for name in DATASET_FALLBACKS:
            p = Path(name)
            if p.exists():
                return p
    raise FileNotFoundError(
        f"Dataset directory not found: {DATASET_DIR!r}. "
        "Change DATASET_DIR in USER SETTINGS."
    )


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


def get_table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    if not table_exists(conn, table):
        return []
    return [r[1] for r in conn.execute(f'PRAGMA table_info({quote_ident(table)});').fetchall()]


def import_csv_to_table(conn: sqlite3.Connection, csv_path: Path, table_name: str, chunksize: int | None) -> None:
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
        chunk.to_sql(table_name, conn, if_exists="replace" if first else "append", index=False)
        total += len(chunk)
        first = False
        print(f"         imported rows: {total:,}", end="\r")
    print(f"         rows: {total:,}")


def optimize_sqlite(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.execute("PRAGMA cache_size = -200000;")
    conn.commit()


def create_indexes(conn: sqlite3.Connection) -> None:
    print("[INDEX] Creating indexes...")
    for sql in INDEX_SQL:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError as exc:
            print(f"[WARN] Index skipped: {exc}")
    conn.commit()


def add_existing_columns(
    select_parts: list[str],
    available_columns: Iterable[str],
    table_alias: str,
    desired_columns: Iterable[str],
    output_prefix: str = "",
) -> None:
    available = set(available_columns)
    for col in desired_columns:
        if col in available:
            alias = f"{output_prefix}{col}"
            select_parts.append(f'{table_alias}.{quote_ident(col)} AS {quote_ident(alias)}')


# ============================================================
# Views
# ============================================================

def create_measurement_summary_view(conn: sqlite3.Connection) -> None:
    if not table_exists(conn, "si_fingerprint_features"):
        print("[VIEW] si_fingerprint_features missing; view_measurement_summary skipped.")
        return

    print("[VIEW] Creating view_measurement_summary...")
    conn.execute("DROP VIEW IF EXISTS view_measurement_summary;")

    f_cols = get_table_columns(conn, "si_fingerprint_features")
    m_cols = get_table_columns(conn, "case_metadata")

    select_parts: list[str] = []
    core = [
        "case_id", "lot_id", "PN_ID", "board_id", "panel_id", "test_point_id",
        "x_mm", "y_mm", "region_id", "section_type", "label", "label_group",
        "point_pattern", "issue_scope", "true_lot_pattern", "true_board_pattern",
        "stage1_overall_status", "stage1_impedance_status", "stage1_loss_status",
        "stage1_overall_risk_score",
    ]
    add_existing_columns(select_parts, f_cols, "f", core)

    vna_loss_cols = [
        "IL_16GHz", "IL_20GHz", "IL_28GHz", "IL_32GHz", "IL_40GHz",
        "IL_slope_20GHz_to_28GHz", "IL_slope_28GHz_to_40GHz",
        "DeltaL_20GHz_dB_per_in", "DeltaL_28GHz_dB_per_in",
        "DeltaL_32GHz_dB_per_in", "DeltaL_40GHz_dB_per_in",
        "loss_baseline_excess_ratio_max",
        "DeltaL_28GHz_dB_per_in_pn_excess_ratio",
        "DeltaL_40GHz_dB_per_in_pn_excess_ratio",
    ]
    add_existing_columns(select_parts, f_cols, "f", vna_loss_cols)

    impedance_shape_cols = [
        "Z_mean", "Z_max_dev", "Z_mean_target_dev_pct", "Z_max_dev_pct",
        "TDR_peak", "TDR_dip", "TDR_peak_abs_ohm", "TDR_peak_abs_pct",
        "TDR_peak_polarity", "TDR_local_peak_width_mm", "TDR_local_peak_area_ohm_mm",
        "TDR_num_regions_above_3pct", "TDR_num_regions_above_7pct",
        "TDR_peak_position_mm", "TDR_energy", "TDR_peak_to_peak",
        "RL_min", "RL_mean", "RL_ripple",
        "notch_depth", "notch_frequency",
        "notch_depth_20GHz_to_40GHz", "notch_frequency_20GHz_to_40GHz", "S21_ripple_20GHz_to_40GHz",
    ]
    add_existing_columns(select_parts, f_cols, "f", impedance_shape_cols)

    metadata_cols = [
        "severity", "defect_type", "defect_position_mm", "defect_length_mm",
        "width_change_pct", "stub_length_mm", "return_path_extra_L_pH",
        "Dk", "Df", "roughness_um", "trace_width_um", "Z0_target", "length_mm",
    ]
    add_existing_columns(select_parts, m_cols, "m", metadata_cols)

    select_sql = ",\n        ".join(select_parts)
    join_sql = ""
    if table_exists(conn, "case_metadata"):
        join_sql = "\nLEFT JOIN case_metadata m ON f.case_id = m.case_id"

    sql = f"""
CREATE VIEW view_measurement_summary AS
SELECT
        {select_sql}
FROM si_fingerprint_features f{join_sql};
"""
    conn.execute(sql)
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM view_measurement_summary;").fetchone()[0]
    print(f"[VIEW] view_measurement_summary rows: {n:,}")


def create_lot_dashboard_view(conn: sqlite3.Connection) -> None:
    if not table_exists(conn, "lot_aggregation_features"):
        print("[VIEW] lot_aggregation_features missing; view_lot_dashboard skipped.")
        return

    print("[VIEW] Creating view_lot_dashboard...")
    conn.execute("DROP VIEW IF EXISTS view_lot_dashboard;")

    # Dynamic fallback: include common columns that exist.
    cols = get_table_columns(conn, "lot_aggregation_features")
    desired = [
        "lot_id", "PN_ID", "true_lot_pattern", "rule_based_lot_diagnosis",
        "total_measurements", "abnormal_rate", "warning_or_worse_rate",
        "out_of_spec_or_worse_rate", "severe_rate", "loss_related_rate",
        "impedance_width_rate", "local_defect_rate", "via_stub_rate", "return_path_rate",
        "avg_Z_mean_target_dev_pct", "avg_Z_max_dev_pct", "avg_loss_excess_ratio",
        "max_loss_excess_ratio", "max_board_abnormal_rate", "max_region_abnormal_rate",
        "fixed_test_point_repeat_rate", "board_concentration_score",
        "regional_concentration_score", "randomness_score",
        "total_panels", "sampled_boards", "test_points_per_board",
    ]
    select_parts: list[str] = []
    add_existing_columns(select_parts, cols, "l", desired)
    select_sql = ",\n        ".join(select_parts)
    conn.execute(f"""
CREATE VIEW view_lot_dashboard AS
SELECT
        {select_sql}
FROM lot_aggregation_features l;
""")
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM view_lot_dashboard;").fetchone()[0]
    print(f"[VIEW] view_lot_dashboard rows: {n:,}")


def create_top_issue_views(conn: sqlite3.Connection) -> None:
    # These helper views are optional and only created when source tables exist.
    if table_exists(conn, "board_aggregation_features"):
        conn.execute("DROP VIEW IF EXISTS view_board_issue_rank;")
        conn.execute("""
CREATE VIEW view_board_issue_rank AS
SELECT *
FROM board_aggregation_features
ORDER BY lot_id, abnormal_rate DESC, board_id;
""")
    if table_exists(conn, "region_aggregation_features"):
        conn.execute("DROP VIEW IF EXISTS view_region_issue_rank;")
        conn.execute("""
CREATE VIEW view_region_issue_rank AS
SELECT *
FROM region_aggregation_features
ORDER BY lot_id, abnormal_rate DESC, region_id;
""")
    if table_exists(conn, "test_point_repeat_features"):
        conn.execute("DROP VIEW IF EXISTS view_test_point_repeat_rank;")
        conn.execute("""
CREATE VIEW view_test_point_repeat_rank AS
SELECT *
FROM test_point_repeat_features
ORDER BY lot_id, abnormal_rate DESC, test_point_id;
""")
    conn.commit()


def create_views(conn: sqlite3.Connection) -> None:
    # Drop known views first for clean reruns.
    for v in [
        "view_measurement_summary", "view_lot_dashboard", "view_board_issue_rank",
        "view_region_issue_rank", "view_test_point_repeat_rank",
    ]:
        conn.execute(f"DROP VIEW IF EXISTS {quote_ident(v)};")
    conn.commit()

    create_measurement_summary_view(conn)
    create_lot_dashboard_view(conn)
    create_top_issue_views(conn)


# ============================================================
# Checks
# ============================================================

def print_table_count(conn: sqlite3.Connection, table: str) -> None:
    if table_exists(conn, table):
        n = conn.execute(f"SELECT COUNT(*) FROM {quote_ident(table)};").fetchone()[0]
        print(f"{table:34s}: {n:,} rows")
    else:
        print(f"{table:34s}: missing")


def print_frequency_range(conn: sqlite3.Connection, table: str, label: str) -> None:
    if not table_exists(conn, table):
        return
    cols = get_table_columns(conn, table)
    if "frequency_GHz" not in cols:
        return
    row = conn.execute(f"SELECT MIN(frequency_GHz), MAX(frequency_GHz), COUNT(*) FROM {quote_ident(table)};").fetchone()
    if row:
        fmin, fmax, n = row
        print(f"{label:34s}: {fmin:.3f} to {fmax:.3f} GHz, rows={n:,}")


def print_basic_checks(conn: sqlite3.Connection) -> None:
    print("\n=== Basic table checks ===")
    for table in [
        "generation_summary", "lot_pattern_counts", "lot_metadata", "board_metadata", "test_point_map",
        "case_metadata", "pn_config", "vna_sparameter_long", "tdr_impedance_long", "deltal_loss_long",
        "si_fingerprint_features", "si_stage1_deviation_features", "si_stage2_shape_features",
        "board_aggregation_features", "region_aggregation_features", "test_point_repeat_features", "lot_aggregation_features",
    ]:
        print_table_count(conn, table)

    print("\n=== Frequency ranges ===")
    print_frequency_range(conn, "vna_sparameter_long", "VNA/S-parameter")
    print_frequency_range(conn, "deltal_loss_long", "Delta-L")

    if table_exists(conn, "lot_metadata"):
        print("\nLot pattern counts:")
        rows = conn.execute("""
            SELECT true_lot_pattern, COUNT(*)
            FROM lot_metadata
            GROUP BY true_lot_pattern
            ORDER BY COUNT(*) DESC;
        """).fetchall()
        for pattern, n in rows:
            print(f"{pattern:34s}: {n:,}")

    if table_exists(conn, "case_metadata"):
        print("\nMeasurement label counts:")
        rows = conn.execute("""
            SELECT label, COUNT(*)
            FROM case_metadata
            GROUP BY label
            ORDER BY COUNT(*) DESC;
        """).fetchall()
        for label, n in rows:
            print(f"{label:34s}: {n:,}")

    for view in ["view_measurement_summary", "view_lot_dashboard", "view_board_issue_rank", "view_region_issue_rank", "view_test_point_repeat_rank"]:
        if view_exists(conn, view):
            n = conn.execute(f"SELECT COUNT(*) FROM {quote_ident(view)};").fetchone()[0]
            print(f"{view:34s}: {n:,} rows")


# ============================================================
# Main
# ============================================================

def main() -> None:
    dataset_dir = resolve_dataset_dir()
    db_path = dataset_dir / DB_NAME

    print("=== SI CSV to SQLite Converter: V4 PN06 Lot-Aware Direct Run ===")
    print(f"Dataset directory: {dataset_dir.resolve()}")
    print(f"SQLite DB path   : {db_path.resolve()}")

    if OVERWRITE_DB_FILE and db_path.exists():
        print(f"[RESET] Removing old DB: {db_path}")
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    try:
        optimize_sqlite(conn)
        for spec in TABLE_SPECS:
            import_csv_to_table(conn, dataset_dir / spec["csv"], spec["table"], spec["chunksize"])
        create_indexes(conn)
        create_views(conn)
        print("[ANALYZE] Optimizing query planner statistics...")
        conn.execute("ANALYZE;")
        conn.commit()
        print_basic_checks(conn)
    finally:
        conn.close()

    print("\nDone.")
    print(f"SQLite database created: {db_path.resolve()}")


if __name__ == "__main__":
    main()
