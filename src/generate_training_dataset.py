"""
generate_training_dataset.py

V3 two-stage simulation-based dataset generator for high-speed PCB
signal-integrity abnormality diagnosis.

This script generates synthetic, physics-inspired data for:
1. VNA / S-parameter curves:
   - S11_dB, S21_dB, phase
2. TDR impedance curves:
   - impedance vs distance/time
3. Simplified Delta-L-like loss extraction:
   - loss_dB_per_in from short/long coupon S21
4. Case metadata:
   - PN, label, stack-up-like parameters, defect parameters

Important:
- This is NOT a full-wave EM solver.
- The objective is to generate controlled, physics-inspired signal fingerprints
  for AI diagnosis pipeline development.
- The curves are designed to resemble engineering tendencies:
  roughness/high-Df -> high-frequency loss increase
  width variation -> impedance shift and return-loss degradation
  local defect -> local TDR spike/dip and S11 ripple
  via stub -> resonance notch
  return path issue -> ripple / broad impedance disturbance

Author: Po-Hung Liu
Developed with AI-assisted coding support.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import warnings
from pandas.errors import PerformanceWarning

warnings.simplefilter(action="ignore", category=PerformanceWarning)


# ============================================================
# 1. Default configuration
# ============================================================

DEFAULT_SEED = 42

# ============================================================
# DIRECT RUN USER SETTINGS
# ============================================================
# You can run this file directly in VS Code.
# Change these values if needed.
RUN_OUTPUT_DIR = "si_simulated_dataset_v3"
RUN_PN_COUNT = 6
RUN_CASES_PER_PN = 1000
RUN_FREQ_POINTS = 401
RUN_TDR_POINTS = 301
RUN_FREQ_MAX_GHZ = 40.0
RUN_SEED = 42
RUN_SAVE_FEATURES = True

DEFAULT_OUTPUT_DIR = "si_simulated_dataset_v3"

# MVP setting:
DEFAULT_PN_COUNT = 6
DEFAULT_CASES_PER_PN = 1000
DEFAULT_FREQ_POINTS = 401
DEFAULT_TDR_POINTS = 301

# Frequency range for VNA / Delta-L.
# 40 GHz is the default practical target for Delta-L 4.0 / 112G-class
# development. Use --freq_max 56 for 224G-class exploratory data.
FREQ_MIN_GHZ = 0.1
FREQ_MAX_GHZ = 40.0

# Frequency-aware SI fingerprint definition.
# These are aligned with common product-generation checkpoints:
# 12.5G: 25G NRZ Nyquist, 14G: 56G PAM4 symbol Nyquist,
# 16G: PCIe 5/6 class, 20G+: post-2025 high-speed PCB screening,
# 28G: 112G PAM4, 32G: PCIe 7 class, 40G: Delta-L 4.0 margin.
KEY_FREQS_GHZ = [1.0, 5.0, 10.0, 12.5, 14.0, 16.0, 20.0, 28.0, 32.0, 40.0]
SLOPE_BANDS_GHZ = [(1.0, 5.0), (5.0, 10.0), (10.0, 16.0), (16.0, 20.0), (20.0, 28.0), (28.0, 40.0)]
NOTCH_BANDS_GHZ = [(0.1, 20.0), (20.0, 40.0)]

# ============================================================
# V3 two-stage diagnostic calibration settings
# ============================================================
# Stage 1 focuses on average / baseline deviation.
# Stage 2 focuses on local shape anomalies such as TDR spikes/dips
# and S21 notch/ripple.
IMPEDANCE_TOLERANCE_PCT = 0.07     # +/-7% impedance tolerance
IMPEDANCE_WARNING_PCT = 0.03       # warning zone before the 7% limit
IMPEDANCE_SEVERE_PCT = 0.10        # strong out-of-spec deviation

# Baseline loss risk is defined relative to the PN golden baseline.
# Delta-L is positive dB/in; higher is worse.
LOSS_WARNING_EXCESS_RATIO = 0.10   # +10% above PN baseline
LOSS_ABNORMAL_EXCESS_RATIO = 0.25  # +25% above PN baseline
LOSS_SEVERE_EXCESS_RATIO = 0.40    # +40% above PN baseline

# High-speed fine-line calibration. PN nominal widths are around 4-6 mil.
# V3 adds near-boundary local defects so the model can learn warning-zone
# behavior rather than only obvious out-of-spec cases.
SENSITIVITY_SEVERITY_LEVELS = ["warning", "mild", "medium", "severe"]

# Case distribution per PN.
# Total must equal cases_per_pn if using default 1000.
DEFAULT_CLASS_COUNTS_PER_PN = {
    "golden_normal": 30,
    "normal_production": 220,
    "roughness_high": 125,
    "high_df": 125,
    "width_variation": 150,
    "local_defect": 125,
    "via_stub": 125,
    "return_path_issue": 100,
}

# Physical constants
C0 = 299_792_458.0  # speed of light, m/s
MU0 = 4.0 * math.pi * 1e-7
SIGMA_CU = 5.8e7


# ============================================================
# 2. PN definitions
# ============================================================

@dataclass
class PNConfig:
    PN_ID: str
    description: str
    line_type: str
    Z0_target: float           # ohm
    length_mm: float
    Dk_nominal: float
    Df_nominal: float
    roughness_nominal_um: float
    trace_width_nominal_um: float
    copper_thickness_nominal_um: float
    velocity_factor: float
    short_coupon_mm: float
    long_coupon_mm: float


def build_default_pns(pn_count: int = 6) -> List[PNConfig]:
    """Build a set of PN configurations with different baselines."""
    base_pns = [
        PNConfig(
            PN_ID="PN01",
            description="Short 50-ohm single-ended line",
            line_type="single_ended",
            Z0_target=50.0,
            length_mm=80.0,
            Dk_nominal=3.45,
            Df_nominal=0.0040,
            roughness_nominal_um=0.6,
            trace_width_nominal_um=100.0,
            copper_thickness_nominal_um=18.0,
            velocity_factor=0.55,
            short_coupon_mm=50.0,
            long_coupon_mm=150.0,
        ),
        PNConfig(
            PN_ID="PN02",
            description="Long 50-ohm single-ended line",
            line_type="single_ended",
            Z0_target=50.0,
            length_mm=160.0,
            Dk_nominal=3.55,
            Df_nominal=0.0045,
            roughness_nominal_um=0.7,
            trace_width_nominal_um=98.0,
            copper_thickness_nominal_um=18.0,
            velocity_factor=0.53,
            short_coupon_mm=70.0,
            long_coupon_mm=220.0,
        ),
        PNConfig(
            PN_ID="PN03",
            description="Low-loss material line",
            line_type="single_ended",
            Z0_target=50.0,
            length_mm=120.0,
            Dk_nominal=3.20,
            Df_nominal=0.0025,
            roughness_nominal_um=0.45,
            trace_width_nominal_um=105.0,
            copper_thickness_nominal_um=18.0,
            velocity_factor=0.58,
            short_coupon_mm=60.0,
            long_coupon_mm=180.0,
        ),
        PNConfig(
            PN_ID="PN04",
            description="Via-rich high-speed line",
            line_type="single_ended",
            Z0_target=50.0,
            length_mm=140.0,
            Dk_nominal=3.50,
            Df_nominal=0.0040,
            roughness_nominal_um=0.65,
            trace_width_nominal_um=100.0,
            copper_thickness_nominal_um=18.0,
            velocity_factor=0.54,
            short_coupon_mm=60.0,
            long_coupon_mm=200.0,
        ),
        PNConfig(
            PN_ID="PN05",
            description="DDR-like 45-ohm single-ended line",
            line_type="single_ended_ddr_like",
            Z0_target=45.0,
            length_mm=90.0,
            Dk_nominal=3.60,
            Df_nominal=0.0042,
            roughness_nominal_um=0.7,
            trace_width_nominal_um=115.0,
            copper_thickness_nominal_um=18.0,
            velocity_factor=0.52,
            short_coupon_mm=50.0,
            long_coupon_mm=150.0,
        ),
        PNConfig(
            PN_ID="PN06",
            description="Return-path-sensitive line",
            line_type="single_ended",
            Z0_target=50.0,
            length_mm=130.0,
            Dk_nominal=3.48,
            Df_nominal=0.0040,
            roughness_nominal_um=0.65,
            trace_width_nominal_um=100.0,
            copper_thickness_nominal_um=18.0,
            velocity_factor=0.54,
            short_coupon_mm=60.0,
            long_coupon_mm=190.0,
        ),
    ]

    if pn_count <= len(base_pns):
        return base_pns[:pn_count]

    # If more PN are requested, extend by perturbing existing PN definitions.
    pns = base_pns.copy()
    rng = np.random.default_rng(DEFAULT_SEED + 999)

    for i in range(len(base_pns) + 1, pn_count + 1):
        template = base_pns[(i - 1) % len(base_pns)]
        scale_len = rng.uniform(0.85, 1.25)
        z0_shift = rng.choice([-5, 0, 5])
        pns.append(
            PNConfig(
                PN_ID=f"PN{i:02d}",
                description=f"Synthetic variant of {template.PN_ID}",
                line_type=template.line_type,
                Z0_target=max(40.0, template.Z0_target + z0_shift),
                length_mm=template.length_mm * scale_len,
                Dk_nominal=template.Dk_nominal + rng.normal(0, 0.08),
                Df_nominal=max(0.0015, template.Df_nominal + rng.normal(0, 0.0005)),
                roughness_nominal_um=max(0.2, template.roughness_nominal_um + rng.normal(0, 0.1)),
                trace_width_nominal_um=max(70, template.trace_width_nominal_um + rng.normal(0, 5)),
                copper_thickness_nominal_um=template.copper_thickness_nominal_um,
                velocity_factor=np.clip(template.velocity_factor + rng.normal(0, 0.02), 0.48, 0.62),
                short_coupon_mm=template.short_coupon_mm * scale_len,
                long_coupon_mm=template.long_coupon_mm * scale_len,
            )
        )
    return pns


# ============================================================
# 3. Utility functions
# ============================================================

def db20(x: np.ndarray | float) -> np.ndarray | float:
    return 20.0 * np.log10(np.maximum(np.asarray(x), 1e-15))


def skin_depth_um(freq_ghz: np.ndarray) -> np.ndarray:
    """Copper skin depth in um."""
    f_hz = freq_ghz * 1e9
    omega = 2.0 * np.pi * f_hz
    delta_m = np.sqrt(2.0 / (omega * MU0 * SIGMA_CU))
    return delta_m * 1e6


def smooth_noise(rng: np.random.Generator, n: int, scale: float, kernel_size: int = 9) -> np.ndarray:
    """Create smooth noise for curve-like behavior."""
    raw = rng.normal(0.0, scale, size=n)
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(raw, kernel, mode="same")


def gaussian(x: np.ndarray, center: float, width: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


def choose_severity(rng: np.random.Generator) -> str:
    return rng.choice(["mild", "medium", "severe"], p=[0.45, 0.35, 0.20])


def choose_sensitivity_severity(rng: np.random.Generator) -> str:
    """V3 severity levels for near-boundary fine-line defects."""
    return rng.choice(SENSITIVITY_SEVERITY_LEVELS, p=[0.25, 0.35, 0.25, 0.15])


def risk_score_from_pct(abs_pct: float) -> int:
    """0 normal, 1 warning, 2 out-of-spec, 3 severe."""
    if pd.isna(abs_pct):
        return 0
    if abs_pct >= IMPEDANCE_SEVERE_PCT:
        return 3
    if abs_pct >= IMPEDANCE_TOLERANCE_PCT:
        return 2
    if abs_pct >= IMPEDANCE_WARNING_PCT:
        return 1
    return 0


def risk_label_from_score(score: int) -> str:
    return {0: "normal", 1: "warning", 2: "out_of_spec", 3: "severe"}.get(int(score), "normal")


def loss_risk_score_from_excess_ratio(excess_ratio: float) -> int:
    """0 normal, 1 warning, 2 abnormal, 3 severe based on PN baseline excess."""
    if pd.isna(excess_ratio):
        return 0
    if excess_ratio >= LOSS_SEVERE_EXCESS_RATIO:
        return 3
    if excess_ratio >= LOSS_ABNORMAL_EXCESS_RATIO:
        return 2
    if excess_ratio >= LOSS_WARNING_EXCESS_RATIO:
        return 1
    return 0


# ============================================================
# 4. Case parameter generation
# ============================================================

def sample_case_parameters(
    rng: np.random.Generator,
    pn: PNConfig,
    label: str,
) -> Dict:
    """Generate physical-ish parameters for one case."""
    severity = "none"
    defect_type = "none"

    # Normal production variation
    Dk = pn.Dk_nominal + rng.normal(0, 0.025)
    Df = pn.Df_nominal + rng.normal(0, 0.0002)
    roughness_um = max(0.1, pn.roughness_nominal_um + rng.normal(0, 0.08))
    trace_width_um = pn.trace_width_nominal_um + rng.normal(0, 1.0)
    copper_thickness_um = pn.copper_thickness_nominal_um + rng.normal(0, 0.5)
    dielectric_thickness_factor = 1.0 + rng.normal(0, 0.01)

    width_change_pct = 0.0
    defect_position_mm = 0.0
    defect_length_mm = 0.0
    stub_length_mm = 0.0
    return_path_extra_L_pH = 0.0

    # Golden samples are intentionally very close to nominal
    if label == "golden_normal":
        Dk = pn.Dk_nominal + rng.normal(0, 0.008)
        Df = pn.Df_nominal + rng.normal(0, 0.00008)
        roughness_um = max(0.1, pn.roughness_nominal_um + rng.normal(0, 0.025))
        trace_width_um = pn.trace_width_nominal_um + rng.normal(0, 0.4)
        copper_thickness_um = pn.copper_thickness_nominal_um + rng.normal(0, 0.15)
        dielectric_thickness_factor = 1.0 + rng.normal(0, 0.003)

    elif label == "normal_production":
        severity = "normal"

    elif label == "roughness_high":
        severity = choose_severity(rng)
        if severity == "mild":
            roughness_um = rng.uniform(1.0, 2.0)
        elif severity == "medium":
            roughness_um = rng.uniform(2.0, 3.5)
        else:
            roughness_um = rng.uniform(3.5, 5.0)

    elif label == "high_df":
        severity = choose_severity(rng)
        if severity == "mild":
            Df = rng.uniform(0.0050, 0.0060)
        elif severity == "medium":
            Df = rng.uniform(0.0060, 0.0080)
        else:
            Df = rng.uniform(0.0080, 0.0120)

    elif label == "width_variation":
        # V3: make global width shifts sensitive to +/-7% impedance tolerance.
        # warning cases are near-boundary process shifts; medium/severe are out-of-spec.
        severity = choose_sensitivity_severity(rng)
        direction = rng.choice(["narrow", "wide"])
        if severity == "warning":
            mag = rng.uniform(0.02, 0.05)
        elif severity == "mild":
            mag = rng.uniform(0.05, 0.075)
        elif severity == "medium":
            mag = rng.uniform(0.075, 0.12)
        else:
            mag = rng.uniform(0.12, 0.18)
        width_change_pct = -100 * mag if direction == "narrow" else 100 * mag
        trace_width_um = pn.trace_width_nominal_um * (1 + width_change_pct / 100.0)
        # dielectric thickness drift can create similar impedance shifts
        dielectric_thickness_factor += rng.normal(0, 0.012)

    elif label == "local_defect":
        # V3: include near-boundary defects scaled for 4-6 mil fine high-speed traces.
        severity = choose_sensitivity_severity(rng)
        defect_type = rng.choice(["dent", "bump", "notch", "scratch"])
        defect_position_mm = rng.uniform(0.20, 0.80) * pn.length_mm
        if severity == "warning":
            defect_length_mm = rng.uniform(0.2, 0.6)
            mag = rng.uniform(0.02, 0.05)
        elif severity == "mild":
            defect_length_mm = rng.uniform(0.5, 1.2)
            mag = rng.uniform(0.05, 0.10)
        elif severity == "medium":
            defect_length_mm = rng.uniform(1.0, 2.5)
            mag = rng.uniform(0.10, 0.18)
        else:
            defect_length_mm = rng.uniform(2.5, 5.0)
            mag = rng.uniform(0.18, 0.30)

        if defect_type in ["dent", "notch", "scratch"]:
            width_change_pct = -100 * mag
        else:
            width_change_pct = 100 * mag

    elif label == "via_stub":
        severity = choose_severity(rng)
        defect_type = "via_stub"
        defect_position_mm = rng.uniform(0.25, 0.75) * pn.length_mm
        if severity == "mild":
            stub_length_mm = rng.uniform(0.2, 0.5)
        elif severity == "medium":
            stub_length_mm = rng.uniform(0.5, 1.0)
        else:
            stub_length_mm = rng.uniform(1.0, 2.0)

    elif label == "return_path_issue":
        severity = choose_severity(rng)
        defect_type = rng.choice(["plane_void", "reference_split", "missing_stitching_via"])
        defect_position_mm = rng.uniform(0.25, 0.75) * pn.length_mm
        if severity == "mild":
            defect_length_mm = rng.uniform(3.0, 8.0)
            return_path_extra_L_pH = rng.uniform(20, 60)
        elif severity == "medium":
            defect_length_mm = rng.uniform(8.0, 18.0)
            return_path_extra_L_pH = rng.uniform(60, 140)
        else:
            defect_length_mm = rng.uniform(18.0, 35.0)
            return_path_extra_L_pH = rng.uniform(140, 300)

    else:
        raise ValueError(f"Unknown label: {label}")

    return {
        "severity": severity,
        "Dk": float(Dk),
        "Df": float(max(Df, 0.0005)),
        "roughness_um": float(max(roughness_um, 0.1)),
        "trace_width_um": float(max(trace_width_um, 20.0)),
        "copper_thickness_um": float(max(copper_thickness_um, 5.0)),
        "dielectric_thickness_factor": float(max(dielectric_thickness_factor, 0.8)),
        "defect_type": defect_type,
        "defect_position_mm": float(defect_position_mm),
        "defect_length_mm": float(defect_length_mm),
        "width_change_pct": float(width_change_pct),
        "stub_length_mm": float(stub_length_mm),
        "return_path_extra_L_pH": float(return_path_extra_L_pH),
    }


# ============================================================
# 5. Simulation models
# ============================================================

def estimate_impedance(
    pn: PNConfig,
    trace_width_um: float,
    Dk: float,
    dielectric_thickness_factor: float,
    width_change_pct: float = 0.0,
) -> float:
    """
    Simplified impedance estimate.

    This is not a precise microstrip/stripline equation.
    It creates realistic directionality:
    - narrower trace -> higher impedance
    - wider trace -> lower impedance
    - higher Dk -> lower impedance
    - thicker dielectric -> higher impedance
    """
    width_ratio = trace_width_um / pn.trace_width_nominal_um
    dk_ratio = Dk / pn.Dk_nominal
    z = pn.Z0_target
    z *= width_ratio ** (-0.45)
    z *= dk_ratio ** (-0.50)
    z *= dielectric_thickness_factor ** (0.55)
    return float(z)


def base_loss_db(
    freq_ghz: np.ndarray,
    length_mm: float,
    Df: float,
    roughness_um: float,
    copper_thickness_um: float,
) -> np.ndarray:
    """
    Generate insertion loss in dB.

    The form is physics-inspired:
    - conductor loss scales roughly with sqrt(f)
    - dielectric loss scales roughly with f
    - roughness increases conductor loss as roughness/skin-depth grows
    """
    length_in = length_mm / 25.4
    delta_um = skin_depth_um(freq_ghz)

    # Roughness correction. Clipped to avoid unrealistic explosion.
    rough_factor = 1.0 + 0.18 * np.clip(roughness_um / np.maximum(delta_um, 1e-6), 0, 8)

    # Copper thickness effect, thin copper slightly increases conductor loss.
    thickness_factor = (18.0 / max(copper_thickness_um, 1.0)) ** 0.15

    conductor_db_per_in = 0.045 * np.sqrt(freq_ghz) * rough_factor * thickness_factor
    dielectric_db_per_in = 1.65 * Df * freq_ghz

    il = -(conductor_db_per_in + dielectric_db_per_in) * length_in
    return il


def reflection_from_impedance(z: float, zref: float = 50.0) -> float:
    """Return reflection coefficient magnitude from impedance mismatch."""
    gamma = abs((z - zref) / (z + zref))
    return float(np.clip(gamma, 1e-5, 0.95))


def simulate_vna(
    rng: np.random.Generator,
    pn: PNConfig,
    params: Dict,
    freq_ghz: np.ndarray,
) -> pd.DataFrame:
    """Simulate S11/S21 frequency response."""
    n = len(freq_ghz)

    z_eff = estimate_impedance(
        pn,
        trace_width_um=params["trace_width_um"],
        Dk=params["Dk"],
        dielectric_thickness_factor=params["dielectric_thickness_factor"],
    )

    s21 = base_loss_db(
        freq_ghz,
        length_mm=pn.length_mm,
        Df=params["Df"],
        roughness_um=params["roughness_um"],
        copper_thickness_um=params["copper_thickness_um"],
    )

    # Width/geometric mismatch affects S11 and mildly affects S21.
    gamma = reflection_from_impedance(z_eff, pn.Z0_target)
    base_s11 = db20(gamma + 0.006 + 0.0004 * freq_ghz)
    base_s11 = np.maximum(base_s11, -45.0)

    # Add small frequency-dependent natural ripple.
    ripple = 0.08 * np.sin(2 * np.pi * freq_ghz / rng.uniform(12, 25) + rng.uniform(0, 2*np.pi))
    s21 = s21 + ripple + smooth_noise(rng, n, 0.025, 11)
    s11 = base_s11 + smooth_noise(rng, n, 0.18, 9)

    label = params["label"]

    # Local defect: ripple and extra small loss, tied to defect length/intensity.
    if label == "local_defect":
        mag = abs(params["width_change_pct"]) / 100.0
        ripple_amp = 0.4 + 2.2 * mag
        period = rng.uniform(5, 16)
        phase = rng.uniform(0, 2*np.pi)
        s21 += -0.15 * mag * freq_ghz / max(float(np.max(freq_ghz)), 1e-9)
        s21 += 0.20 * ripple_amp * np.sin(2 * np.pi * freq_ghz / period + phase)
        s11 += 2.0 + 7.0 * mag + 2.0 * np.sin(2 * np.pi * freq_ghz / period + phase)

    # Via stub: resonant notch in S21 and S11 degradation.
    if label == "via_stub":
        stub = max(params["stub_length_mm"], 0.05)
        eps_eff = max(params["Dk"] * 0.75, 1.5)
        f_res_hz = C0 / (4 * stub * 1e-3 * math.sqrt(eps_eff))
        f_res_ghz = np.clip(f_res_hz / 1e9, 2.0, 38.0)
        notch_width = rng.uniform(1.0, 4.0) * (1 + 0.2 * stub)
        notch_depth = np.clip(2.0 + 4.5 * stub + rng.normal(0, 0.35), 1.2, 12.0)
        s21 += gaussian(freq_ghz, f_res_ghz, notch_width, -notch_depth)
        s11 += gaussian(freq_ghz, f_res_ghz, notch_width * 1.2, notch_depth * 1.2)

    # Return path: broad ripple and high-frequency degradation.
    if label == "return_path_issue":
        L_pH = params["return_path_extra_L_pH"]
        amp = np.clip(L_pH / 80.0, 0.2, 4.0)
        period = rng.uniform(7, 18)
        phase = rng.uniform(0, 2*np.pi)
        broad_center = rng.uniform(15, 32)
        s21 += -0.03 * amp * freq_ghz
        s21 += 0.35 * amp * np.sin(2 * np.pi * freq_ghz / period + phase)
        s11 += 1.5 * amp + 1.2 * amp * np.sin(2 * np.pi * freq_ghz / period + phase)
        s11 += gaussian(freq_ghz, broad_center, rng.uniform(3.0, 8.0), 2.0 * amp)

    # Roughness/high-Df are already reflected in base loss.
    # To avoid impossible values:
    s21 = np.minimum(s21, -0.01)
    s11 = np.clip(s11, -55.0, -3.0)

    # Phase: simple delay model plus perturbations.
    # Phase wraps into [-180, 180].
    vf = pn.velocity_factor * math.sqrt(pn.Dk_nominal / max(params["Dk"], 1.0))
    delay_s = (pn.length_mm * 1e-3) / (C0 * vf)
    phase_s21 = -360.0 * (freq_ghz * 1e9) * delay_s
    phase_s21 = ((phase_s21 + 180) % 360) - 180
    phase_s21 += smooth_noise(rng, n, 0.8, 7)

    phase_s11 = -0.55 * phase_s21 + smooth_noise(rng, n, 1.2, 7)
    phase_s11 = ((phase_s11 + 180) % 360) - 180

    return pd.DataFrame({
        "frequency_GHz": freq_ghz,
        "S11_dB": s11,
        "S21_dB": s21,
        "S11_phase_deg": phase_s11,
        "S21_phase_deg": phase_s21,
        "Z_est_ohm": z_eff,
    })


def simulate_tdr(
    rng: np.random.Generator,
    pn: PNConfig,
    params: Dict,
    distances_mm: np.ndarray,
) -> pd.DataFrame:
    """Simulate TDR impedance vs distance."""
    n = len(distances_mm)

    z_eff = estimate_impedance(
        pn,
        trace_width_um=params["trace_width_um"],
        Dk=params["Dk"],
        dielectric_thickness_factor=params["dielectric_thickness_factor"],
    )

    z = np.full(n, z_eff)

    # Normal distributed process roughness / TDR measurement noise
    z += smooth_noise(rng, n, 0.12, 9)

    label = params["label"]

    # For width variation, the impedance shift is broad/global.
    if label == "width_variation":
        # Add mild end transition to avoid perfectly flat curve.
        z += smooth_noise(rng, n, 0.15, 13)

    # Local defect: narrow spike or dip.
    if label == "local_defect":
        pos = params["defect_position_mm"]
        length = max(params["defect_length_mm"], 0.2)
        mag = abs(params["width_change_pct"]) / 100.0

        # Dent/notch/scratch -> narrower -> higher Z.
        # Bump -> wider -> lower Z.
        sign = 1.0 if params["width_change_pct"] < 0 else -1.0
        amp = sign * (0.5 + 37.0 * mag)
        width = max(length / 2.0, 0.4)
        z += gaussian(distances_mm, pos, width, amp)

    # Via stub: local discontinuity at via location.
    if label == "via_stub":
        pos = params["defect_position_mm"]
        stub = max(params["stub_length_mm"], 0.1)
        amp = np.clip(2.0 + 6.0 * stub, 2.0, 15.0)
        z += gaussian(distances_mm, pos, 1.2 + stub, amp)
        # small adjacent dip to mimic complex via transition
        z += gaussian(distances_mm, pos + 2.0 + stub, 1.5 + stub, -0.35 * amp)

    # Return path issue: broad bump / irregular area.
    if label == "return_path_issue":
        pos = params["defect_position_mm"]
        length = max(params["defect_length_mm"], 3.0)
        L_pH = params["return_path_extra_L_pH"]
        amp = np.clip(2.0 + L_pH / 35.0, 2.5, 13.0)
        z += gaussian(distances_mm, pos, length / 2.5, amp)
        z += 0.7 * np.sin((distances_mm - pos) / max(length, 1.0) * 2 * np.pi) * gaussian(
            distances_mm, pos, length / 2.0, 1.0
        )

    # Roughness/high-Df do not strongly alter TDR impedance.
    # Add launch/end artifacts mildly.
    z += gaussian(distances_mm, 0.0, 1.0, rng.normal(0.5, 0.3))
    z += gaussian(distances_mm, pn.length_mm, 1.0, rng.normal(-0.5, 0.3))

    # Time conversion, round-trip style approximation:
    # TDR time to distance depends on propagation speed. Use one-way equivalent for readability.
    vf = pn.velocity_factor * math.sqrt(pn.Dk_nominal / max(params["Dk"], 1.0))
    time_ps = distances_mm * 1e-3 / (C0 * vf) * 1e12

    return pd.DataFrame({
        "distance_mm": distances_mm,
        "time_ps": time_ps,
        "impedance_ohm": z,
    })


def simulate_deltal(
    rng: np.random.Generator,
    pn: PNConfig,
    params: Dict,
    freq_ghz: np.ndarray,
) -> pd.DataFrame:
    """
    Simplified Delta-L-like loss extraction.

    It simulates short and long coupons sharing the same material/process state,
    then estimates loss per inch:
        (IL_long - IL_short) / (L_long - L_short)
    Because insertion loss values are negative dB, loss_dB_per_in is reported
    as positive magnitude.
    """
    short_len = pn.short_coupon_mm
    long_len = pn.long_coupon_mm

    il_short = base_loss_db(
        freq_ghz, short_len, params["Df"], params["roughness_um"], params["copper_thickness_um"]
    )
    il_long = base_loss_db(
        freq_ghz, long_len, params["Df"], params["roughness_um"], params["copper_thickness_um"]
    )

    label = params["label"]

    # Add similar anomalies to coupons if relevant, but keep Delta-L mainly loss-focused.
    if label == "via_stub":
        stub = max(params["stub_length_mm"], 0.05)
        eps_eff = max(params["Dk"] * 0.75, 1.5)
        f_res_ghz = np.clip(C0 / (4 * stub * 1e-3 * math.sqrt(eps_eff)) / 1e9, 2.0, 38.0)
        notch_width = 2.5
        notch_depth = np.clip(0.8 + 2.5 * stub, 0.5, 6.0)
        il_long += gaussian(freq_ghz, f_res_ghz, notch_width, -notch_depth)
        il_short += gaussian(freq_ghz, f_res_ghz, notch_width, -0.35 * notch_depth)

    if label == "return_path_issue":
        amp = np.clip(params["return_path_extra_L_pH"] / 120.0, 0.1, 2.5)
        phase = rng.uniform(0, 2*np.pi)
        ripple = 0.15 * amp * np.sin(2 * np.pi * freq_ghz / rng.uniform(8, 18) + phase)
        il_long += ripple - 0.02 * amp * freq_ghz
        il_short += 0.4 * ripple - 0.006 * amp * freq_ghz

    # Measurement noise
    il_short += smooth_noise(rng, len(freq_ghz), 0.015, 7)
    il_long += smooth_noise(rng, len(freq_ghz), 0.020, 7)

    length_delta_in = (long_len - short_len) / 25.4
    loss_dB_per_in = (np.abs(il_long) - np.abs(il_short)) / length_delta_in
    loss_dB_per_in = np.maximum(loss_dB_per_in, 0.0)

    return pd.DataFrame({
        "frequency_GHz": freq_ghz,
        "IL_short_dB": il_short,
        "IL_long_dB": il_long,
        "length_short_mm": short_len,
        "length_long_mm": long_len,
        "loss_dB_per_in": loss_dB_per_in,
    })


# ============================================================
# 6. Feature extraction helper
# ============================================================

def _freq_label(freq_ghz: float) -> str:
    """Convert a frequency value into a stable feature-name token."""
    if float(freq_ghz).is_integer():
        return f"{int(freq_ghz)}GHz"
    return f"{str(freq_ghz).replace('.', 'p')}GHz"


def safe_interp_at(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    """
    Interpolate only when the requested frequency is inside the measured range.

    This avoids silently using the final measured point as a fake high-frequency
    result when, for example, the data only extends to 20 GHz but the feature
    table asks for 28/32/40 GHz.
    """
    if len(x) == 0 or x0 < float(np.min(x)) or x0 > float(np.max(x)):
        return float("nan")
    return float(np.interp(x0, x, y))


def slope_between(x: np.ndarray, y: np.ndarray, f1: float, f2: float) -> float:
    y1 = safe_interp_at(x, y, f1)
    y2 = safe_interp_at(x, y, f2)
    if np.isnan(y1) or np.isnan(y2) or f2 == f1:
        return float("nan")
    return float((y2 - y1) / (f2 - f1))


def band_notch_and_ripple(
    f: np.ndarray,
    y_db: np.ndarray,
    f_low: float,
    f_high: float,
) -> Tuple[float, float, float]:
    """Return notch depth, notch frequency, and ripple amplitude inside a band."""
    mask = (f >= f_low) & (f <= f_high)
    if mask.sum() < 7:
        return float("nan"), float("nan"), float("nan")

    fb = f[mask]
    yb = y_db[mask]
    kernel_size = max(5, min(21, int(mask.sum() // 8) * 2 + 1))
    kernel = np.ones(kernel_size) / kernel_size
    smooth = np.convolve(yb, kernel, mode="same")
    residual = yb - smooth
    idx = int(np.argmin(residual))
    notch_depth = float(-residual[idx])
    notch_freq = float(fb[idx])
    ripple_amp = float(np.percentile(residual, 95) - np.percentile(residual, 5))
    return notch_depth, notch_freq, ripple_amp


def tdr_local_shape_features(
    dist: np.ndarray,
    z: np.ndarray,
    z0_target: float,
) -> Dict[str, float]:
    """Extract local shape features for Stage 2 defect diagnosis."""
    z_dev = z - z0_target
    abs_dev = np.abs(z_dev)
    n = len(z_dev)
    if n == 0:
        return {}

    idx = int(np.argmax(abs_dev))
    peak_abs = float(abs_dev[idx])
    peak_signed = float(z_dev[idx])
    polarity = 1.0 if peak_signed >= 0 else -1.0

    # Estimate local width by full-width-at-half-maximum around the strongest deviation.
    half = 0.5 * peak_abs
    left = idx
    while left > 0 and abs_dev[left] >= half:
        left -= 1
    right = idx
    while right < n - 1 and abs_dev[right] >= half:
        right += 1
    peak_width_mm = float(max(dist[right] - dist[left], 0.0))

    # Area around the peak region.
    local_area = float(np.trapz(abs_dev[left:right + 1], dist[left:right + 1])) if right > left else 0.0

    warning_thr = IMPEDANCE_WARNING_PCT * z0_target
    spec_thr = IMPEDANCE_TOLERANCE_PCT * z0_target

    def count_regions(thr: float) -> int:
        mask = abs_dev >= thr
        if not mask.any():
            return 0
        # count rising edges in the boolean mask
        return int(np.sum(mask & ~np.r_[False, mask[:-1]]))

    return {
        "TDR_peak_abs_ohm": peak_abs,
        "TDR_peak_polarity": polarity,
        "TDR_local_peak_width_mm": peak_width_mm,
        "TDR_local_peak_area_ohm_mm": local_area,
        "TDR_num_regions_above_3pct": count_regions(warning_thr),
        "TDR_num_regions_above_7pct": count_regions(spec_thr),
        "TDR_warning_threshold_ohm": float(warning_thr),
        "TDR_spec_threshold_ohm": float(spec_thr),
    }


def extract_features_from_curves(
    case_id: str,
    pn_id: str,
    label: str,
    vna_df: pd.DataFrame,
    tdr_df: pd.DataFrame,
    deltal_df: pd.DataFrame,
    z0_target: float,
    key_freqs_ghz: List[float] | None = None,
    slope_bands_ghz: List[Tuple[float, float]] | None = None,
    notch_bands_ghz: List[Tuple[float, float]] | None = None,
) -> Dict:
    """Extract case-level features from VNA/TDR/Delta-L curves.

    V2 upgrade:
    - uses a configurable key-frequency table rather than only 5/10/20/40 GHz;
    - adds product-generation checkpoints such as 12.5/14/16/28/32 GHz;
    - adds segmented high-frequency slopes;
    - adds notch/ripple features below and above 20 GHz;
    - returns NaN when a requested frequency is outside the measured range.
    """
    key_freqs_ghz = key_freqs_ghz or KEY_FREQS_GHZ
    slope_bands_ghz = slope_bands_ghz or SLOPE_BANDS_GHZ
    notch_bands_ghz = notch_bands_ghz or NOTCH_BANDS_GHZ

    f = vna_df["frequency_GHz"].to_numpy()
    s21 = vna_df["S21_dB"].to_numpy()
    s11 = vna_df["S11_dB"].to_numpy()
    z = tdr_df["impedance_ohm"].to_numpy()
    dist = tdr_df["distance_mm"].to_numpy()
    dl_f = deltal_df["frequency_GHz"].to_numpy()
    dl = deltal_df["loss_dB_per_in"].to_numpy()

    features: Dict[str, float | str] = {
        "case_id": case_id,
        "PN_ID": pn_id,
        "label": label,
        "freq_min_GHz": float(np.min(f)),
        "freq_max_GHz": float(np.max(f)),
    }

    # Frequency-point insertion loss and Delta-L features.
    for freq in key_freqs_ghz:
        token = _freq_label(freq)
        features[f"IL_{token}"] = safe_interp_at(f, s21, freq)
        features[f"DeltaL_{token}_dB_per_in"] = safe_interp_at(dl_f, dl, freq)

    # Backward-compatible aliases used by the previous pipeline.
    for freq in [5.0, 10.0, 20.0, 40.0]:
        token = _freq_label(freq)
        features[f"IL_{int(freq)}GHz"] = features.get(f"IL_{token}", safe_interp_at(f, s21, freq))
        features[f"DeltaL_{int(freq)}GHz_dB_per_in"] = features.get(
            f"DeltaL_{token}_dB_per_in", safe_interp_at(dl_f, dl, freq)
        )

    # Segmented loss slopes. S21 values are negative dB, so a more negative slope
    # indicates faster loss growth with frequency.
    for f1, f2 in slope_bands_ghz:
        b = f"{_freq_label(f1)}_to_{_freq_label(f2)}"
        features[f"IL_slope_{b}"] = slope_between(f, s21, f1, f2)
        features[f"DeltaL_slope_{b}"] = slope_between(dl_f, dl, f1, f2)

    # Backward-compatible high-frequency aliases.
    features["IL_slope_high"] = slope_between(f, s21, 20.0, 40.0)
    features["DeltaL_slope_high"] = slope_between(dl_f, dl, 20.0, 40.0)

    # Return loss features.
    features["RL_min"] = float(np.max(s11))  # less negative = worse
    features["RL_mean"] = float(np.mean(s11))
    features["RL_ripple"] = float(np.percentile(s11, 95) - np.percentile(s11, 5))

    # Global notch features in S21.
    kernel_size = max(5, min(21, len(s21) // 15 * 2 + 1))
    kernel = np.ones(kernel_size) / kernel_size
    smooth_s21 = np.convolve(s21, kernel, mode="same")
    residual = s21 - smooth_s21
    notch_idx = int(np.argmin(residual))
    features["notch_depth"] = float(-residual[notch_idx])
    features["notch_frequency"] = float(f[notch_idx])

    # Band-specific notch/ripple. This is important because post-20 GHz behavior
    # is now a separate diagnostic region.
    for f1, f2 in notch_bands_ghz:
        b = f"{_freq_label(f1)}_to_{_freq_label(f2)}"
        depth, freq_at_notch, ripple = band_notch_and_ripple(f, s21, f1, f2)
        features[f"notch_depth_{b}"] = depth
        features[f"notch_frequency_{b}"] = freq_at_notch
        features[f"S21_ripple_{b}"] = ripple

    # TDR features.
    z_dev = z - z0_target
    features["Z_mean"] = float(np.mean(z))
    features["Z_max_dev"] = float(np.max(np.abs(z_dev)))
    features["TDR_peak"] = float(np.max(z_dev))
    features["TDR_dip"] = float(np.min(z_dev))
    peak_idx = int(np.argmax(np.abs(z_dev)))
    features["TDR_peak_position_mm"] = float(dist[peak_idx])
    features["TDR_energy"] = float(np.mean(z_dev ** 2))
    features["TDR_peak_to_peak"] = float(np.max(z) - np.min(z))

    # V3 Stage 2 local-shape features.
    features.update(tdr_local_shape_features(dist, z, z0_target))

    return features


# ============================================================
# 7. Main generation function
# ============================================================

def compute_nominal_theory_tables(pns: List[PNConfig]) -> Dict[str, Dict[str, float]]:
    """Compute nominal theory values for Stage 1 target-deviation features."""
    out: Dict[str, Dict[str, float]] = {}
    for pn in pns:
        row: Dict[str, float] = {}
        for freq in KEY_FREQS_GHZ:
            token = _freq_label(freq)
            f_arr = np.array([freq], dtype=float)
            il = float(base_loss_db(
                f_arr, pn.length_mm, pn.Df_nominal,
                pn.roughness_nominal_um, pn.copper_thickness_nominal_um,
            )[0])
            short_il = float(base_loss_db(
                f_arr, pn.short_coupon_mm, pn.Df_nominal,
                pn.roughness_nominal_um, pn.copper_thickness_nominal_um,
            )[0])
            long_il = float(base_loss_db(
                f_arr, pn.long_coupon_mm, pn.Df_nominal,
                pn.roughness_nominal_um, pn.copper_thickness_nominal_um,
            )[0])
            length_delta_in = max((pn.long_coupon_mm - pn.short_coupon_mm) / 25.4, 1e-6)
            dl = (abs(long_il) - abs(short_il)) / length_delta_in
            row[f"IL_{token}_theory_dB"] = il
            row[f"DeltaL_{token}_theory_dB_per_in"] = dl
        out[pn.PN_ID] = row
    return out


def add_v3_stage_features(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    pns: List[PNConfig],
) -> pd.DataFrame:
    """Add V3 Stage 1 baseline/theory-deviation and risk features."""
    out = features.copy()
    meta_cols = [
        "case_id", "Z0_target", "length_mm", "severity", "defect_type",
        "width_change_pct", "defect_length_mm", "defect_position_mm",
    ]
    meta_cols = [c for c in meta_cols if c in metadata.columns]
    out = out.merge(metadata[meta_cols], on="case_id", how="left", suffixes=("", "_meta"))

    # Impedance theory / tolerance deviation.
    z0 = out["Z0_target"].replace(0, np.nan)
    out["Z_mean_target_dev_ohm"] = out["Z_mean"] - z0
    out["Z_mean_target_dev_pct"] = out["Z_mean_target_dev_ohm"] / z0
    out["Z_max_dev_pct"] = out["Z_max_dev"] / z0
    out["TDR_peak_pct"] = out["TDR_peak"] / z0
    out["TDR_dip_pct"] = out["TDR_dip"] / z0
    out["TDR_peak_abs_pct"] = out["TDR_peak_abs_ohm"] / z0

    out["impedance_mean_risk_score"] = out["Z_mean_target_dev_pct"].abs().map(risk_score_from_pct)
    out["impedance_local_risk_score"] = out["Z_max_dev_pct"].abs().map(risk_score_from_pct)
    out["impedance_mean_risk"] = out["impedance_mean_risk_score"].map(risk_label_from_score)
    out["impedance_local_risk"] = out["impedance_local_risk_score"].map(risk_label_from_score)

    # Add nominal theoretical insertion loss / Delta-L deviations.
    theory = compute_nominal_theory_tables(pns)
    for pn_id, vals in theory.items():
        idx = out["PN_ID"] == pn_id
        for k, v in vals.items():
            out.loc[idx, k] = v

    for freq in KEY_FREQS_GHZ:
        token = _freq_label(freq)
        il_col = f"IL_{token}"
        dl_col = f"DeltaL_{token}_dB_per_in"
        il_theory_col = f"IL_{token}_theory_dB"
        dl_theory_col = f"DeltaL_{token}_theory_dB_per_in"
        if il_col in out.columns:
            # Negative values: more negative means worse.
            out[f"{il_col}_theory_dev_dB"] = out[il_col] - out[il_theory_col]
        if dl_col in out.columns:
            denom = out[dl_theory_col].replace(0, np.nan)
            out[f"{dl_col}_theory_excess_ratio"] = (out[dl_col] - denom) / denom

    # Robust PN/golden baseline deviations.
    baseline_source = out[out["label"] == "golden_normal"].copy()
    if baseline_source.empty:
        baseline_source = out[out["label"] == "normal_production"].copy()

    baseline_cols = []
    for c in out.columns:
        if c.startswith("IL_") and c.endswith("GHz"):
            baseline_cols.append(c)
        if c.startswith("DeltaL_") and c.endswith("dB_per_in"):
            baseline_cols.append(c)
    baseline_cols += [
        "Z_mean", "Z_max_dev", "TDR_peak_abs_ohm", "TDR_energy",
        "RL_min", "RL_mean", "RL_ripple", "notch_depth", "S21_ripple_20GHz_to_40GHz",
    ]
    baseline_cols = [c for c in dict.fromkeys(baseline_cols) if c in out.columns and pd.api.types.is_numeric_dtype(out[c])]

    for col in baseline_cols:
        med = baseline_source.groupby("PN_ID")[col].median()
        mad = baseline_source.groupby("PN_ID")[col].apply(lambda s: np.nanmedian(np.abs(s - np.nanmedian(s))))
        out[f"{col}_pn_baseline_median"] = out["PN_ID"].map(med)
        out[f"{col}_pn_baseline_dev"] = out[col] - out[f"{col}_pn_baseline_median"]
        denom = out["PN_ID"].map(mad).replace(0, np.nan) * 1.4826
        out[f"{col}_pn_robust_z"] = out[f"{col}_pn_baseline_dev"] / denom

    # Stage 1 loss risk uses Delta-L at 20/28/32/40 GHz against PN golden baseline.
    loss_ratio_cols = []
    for freq in [20.0, 28.0, 32.0, 40.0]:
        token = _freq_label(freq)
        col = f"DeltaL_{token}_dB_per_in"
        base = f"{col}_pn_baseline_median"
        if col in out.columns and base in out.columns:
            ratio_col = f"{col}_pn_excess_ratio"
            out[ratio_col] = (out[col] - out[base]) / out[base].replace(0, np.nan)
            loss_ratio_cols.append(ratio_col)

    if loss_ratio_cols:
        out["loss_baseline_excess_ratio_max"] = out[loss_ratio_cols].max(axis=1)
        out["loss_baseline_risk_score"] = out["loss_baseline_excess_ratio_max"].map(loss_risk_score_from_excess_ratio)
        out["loss_baseline_risk"] = out["loss_baseline_risk_score"].map(risk_label_from_score)
    else:
        out["loss_baseline_excess_ratio_max"] = np.nan
        out["loss_baseline_risk_score"] = 0
        out["loss_baseline_risk"] = "normal"

    # Two-stage target labels for later model training.
    out["stage1_impedance_status"] = out[["impedance_mean_risk_score", "impedance_local_risk_score"]].max(axis=1).map(risk_label_from_score)
    out["stage1_loss_status"] = out["loss_baseline_risk_score"].map(risk_label_from_score)
    out["stage1_overall_risk_score"] = out[["impedance_mean_risk_score", "impedance_local_risk_score", "loss_baseline_risk_score"]].max(axis=1)
    out["stage1_overall_status"] = out["stage1_overall_risk_score"].map(risk_label_from_score)

    return out


def validate_class_counts(class_counts: Dict[str, int], cases_per_pn: int) -> None:
    total = sum(class_counts.values())
    if total != cases_per_pn:
        raise ValueError(
            f"class_counts sum to {total}, but cases_per_pn is {cases_per_pn}. "
            "Please adjust class counts or cases_per_pn."
        )


def generate_dataset(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    pn_count: int = DEFAULT_PN_COUNT,
    cases_per_pn: int = DEFAULT_CASES_PER_PN,
    freq_points: int = DEFAULT_FREQ_POINTS,
    tdr_points: int = DEFAULT_TDR_POINTS,
    freq_max_ghz: float = FREQ_MAX_GHZ,
    seed: int = DEFAULT_SEED,
    save_features: bool = True,
) -> None:
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    class_counts = DEFAULT_CLASS_COUNTS_PER_PN.copy()

    # If user changes cases_per_pn, scale default class counts proportionally.
    if cases_per_pn != DEFAULT_CASES_PER_PN:
        labels = list(class_counts.keys())
        ratios = np.array([class_counts[k] for k in labels], dtype=float)
        ratios = ratios / ratios.sum()
        scaled = np.floor(ratios * cases_per_pn).astype(int)

        # Adjust rounding difference.
        diff = cases_per_pn - scaled.sum()
        for i in range(diff):
            scaled[i % len(scaled)] += 1

        class_counts = {label: int(count) for label, count in zip(labels, scaled)}

    validate_class_counts(class_counts, cases_per_pn)

    pns = build_default_pns(pn_count)

    freq_ghz = np.linspace(FREQ_MIN_GHZ, freq_max_ghz, freq_points)

    metadata_rows = []
    vna_chunks = []
    tdr_chunks = []
    deltal_chunks = []
    feature_rows = []

    case_index = 0

    for pn in pns:
        distances_mm = np.linspace(0.0, pn.length_mm, tdr_points)

        for label, count in class_counts.items():
            for local_idx in range(count):
                case_index += 1
                case_id = f"{pn.PN_ID}_C{case_index:06d}"

                params = sample_case_parameters(rng, pn, label)
                params["label"] = label

                # Simulate curves
                vna_df = simulate_vna(rng, pn, params, freq_ghz)
                tdr_df = simulate_tdr(rng, pn, params, distances_mm)
                deltal_df = simulate_deltal(rng, pn, params, freq_ghz)

                # Metadata
                meta = {
                    "case_id": case_id,
                    "PN_ID": pn.PN_ID,
                    "PN_description": pn.description,
                    "label": label,
                    "line_type": pn.line_type,
                    "Z0_target": pn.Z0_target,
                    "length_mm": pn.length_mm,
                    "Dk_nominal": pn.Dk_nominal,
                    "Df_nominal": pn.Df_nominal,
                    "roughness_nominal_um": pn.roughness_nominal_um,
                    "trace_width_nominal_um": pn.trace_width_nominal_um,
                    "copper_thickness_nominal_um": pn.copper_thickness_nominal_um,
                    "short_coupon_mm": pn.short_coupon_mm,
                    "long_coupon_mm": pn.long_coupon_mm,
                    **params,
                }
                metadata_rows.append(meta)

                # Add IDs to long tables.
                for df in (vna_df, tdr_df, deltal_df):
                    df.insert(0, "label", label)
                    df.insert(0, "PN_ID", pn.PN_ID)
                    df.insert(0, "case_id", case_id)

                vna_chunks.append(vna_df.drop(columns=["Z_est_ohm"], errors="ignore"))
                tdr_chunks.append(tdr_df)
                deltal_chunks.append(deltal_df)

                if save_features:
                    feature_rows.append(
                        extract_features_from_curves(
                            case_id=case_id,
                            pn_id=pn.PN_ID,
                            label=label,
                            vna_df=vna_df,
                            tdr_df=tdr_df,
                            deltal_df=deltal_df,
                            z0_target=pn.Z0_target,
                        )
                    )

    # Write files
    metadata = pd.DataFrame(metadata_rows)
    vna_long = pd.concat(vna_chunks, ignore_index=True)
    tdr_long = pd.concat(tdr_chunks, ignore_index=True)
    deltal_long = pd.concat(deltal_chunks, ignore_index=True)

    metadata.to_csv(raw_dir / "case_metadata.csv", index=False)
    vna_long.to_csv(raw_dir / "vna_sparameter_long.csv", index=False)
    tdr_long.to_csv(raw_dir / "tdr_impedance_long.csv", index=False)
    deltal_long.to_csv(raw_dir / "deltal_loss_long.csv", index=False)

    if save_features:
        features = pd.DataFrame(feature_rows)
        features = add_v3_stage_features(features, metadata, pns)
        features.to_csv(processed_dir / "si_fingerprint_features.csv", index=False)
        # Convenience stage-specific tables for the two-stage V3 workflow.
        stage1_cols = [c for c in features.columns if (
            c in {"case_id", "PN_ID", "label", "severity", "defect_type", "stage1_overall_status",
                  "stage1_impedance_status", "stage1_loss_status", "stage1_overall_risk_score"}
            or c.endswith("_target_dev_pct")
            or c.endswith("_target_dev_ohm")
            or c.endswith("_pn_robust_z")
            or c.endswith("_pn_baseline_dev")
            or "risk" in c
            or "theory_dev" in c
            or "theory_excess" in c
        )]
        stage2_cols = [c for c in features.columns if (
            c in {"case_id", "PN_ID", "label", "severity", "defect_type", "defect_position_mm", "defect_length_mm"}
            or c.startswith("TDR_")
            or c.startswith("notch_")
            or c.startswith("S21_ripple")
            or c.startswith("RL_")
        )]
        features[stage1_cols].to_csv(processed_dir / "si_stage1_deviation_features.csv", index=False)
        features[stage2_cols].to_csv(processed_dir / "si_stage2_shape_features.csv", index=False)

    # Write PN config and generation summary.
    pn_df = pd.DataFrame([asdict(pn) for pn in pns])
    pn_df.to_csv(raw_dir / "pn_config.csv", index=False)

    summary_rows = []
    summary_rows.append({"item": "seed", "value": seed})
    summary_rows.append({"item": "pn_count", "value": pn_count})
    summary_rows.append({"item": "cases_per_pn", "value": cases_per_pn})
    summary_rows.append({"item": "total_cases", "value": pn_count * cases_per_pn})
    summary_rows.append({"item": "freq_points", "value": freq_points})
    summary_rows.append({"item": "freq_min_GHz", "value": FREQ_MIN_GHZ})
    summary_rows.append({"item": "freq_max_GHz", "value": freq_max_ghz})
    summary_rows.append({"item": "key_freqs_GHz", "value": ";".join(str(x) for x in KEY_FREQS_GHZ)})
    summary_rows.append({"item": "impedance_tolerance_pct", "value": IMPEDANCE_TOLERANCE_PCT})
    summary_rows.append({"item": "impedance_warning_pct", "value": IMPEDANCE_WARNING_PCT})
    summary_rows.append({"item": "loss_warning_excess_ratio", "value": LOSS_WARNING_EXCESS_RATIO})
    summary_rows.append({"item": "loss_abnormal_excess_ratio", "value": LOSS_ABNORMAL_EXCESS_RATIO})
    summary_rows.append({"item": "v3_local_defect_warning_width_change_pct", "value": "2-5"})
    summary_rows.append({"item": "v3_local_defect_mild_width_change_pct", "value": "5-10"})
    summary_rows.append({"item": "v3_local_defect_medium_width_change_pct", "value": "10-18"})
    summary_rows.append({"item": "v3_local_defect_severe_width_change_pct", "value": "18-30"})
    summary_rows.append({"item": "tdr_points", "value": tdr_points})
    summary_rows.append({"item": "vna_rows", "value": len(vna_long)})
    summary_rows.append({"item": "tdr_rows", "value": len(tdr_long)})
    summary_rows.append({"item": "deltal_rows", "value": len(deltal_long)})
    for label, count in class_counts.items():
        summary_rows.append({"item": f"class_count_per_PN::{label}", "value": count})
        summary_rows.append({"item": f"class_count_total::{label}", "value": count * pn_count})

    pd.DataFrame(summary_rows).to_csv(output_dir / "generation_summary.csv", index=False)

    print("Dataset generated successfully.")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Total cases: {pn_count * cases_per_pn}")
    print(f"VNA rows: {len(vna_long):,}")
    print(f"TDR rows: {len(tdr_long):,}")
    print(f"Delta-L rows: {len(deltal_long):,}")
    if save_features:
        print(f"Feature rows: {len(feature_rows):,}")


# ============================================================
# 8. CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic VNA/TDR/Delta-L data for high-speed PCB SI abnormality diagnosis."
    )
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pn_count", type=int, default=DEFAULT_PN_COUNT)
    parser.add_argument("--cases_per_pn", type=int, default=DEFAULT_CASES_PER_PN)
    parser.add_argument("--freq_points", type=int, default=DEFAULT_FREQ_POINTS)
    parser.add_argument("--freq_max", type=float, default=FREQ_MAX_GHZ, help="Maximum VNA/Delta-L frequency in GHz. Use 40 by default, or 56 for 224G exploratory data.")
    parser.add_argument("--tdr_points", type=int, default=DEFAULT_TDR_POINTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no_features", action="store_true", help="Do not generate processed feature table.")
    return parser.parse_args()


def main() -> None:
    print("=== V3 Two-Stage SI Simulation Data Generator ===")
    print(f"RUN_OUTPUT_DIR     : {RUN_OUTPUT_DIR}")
    print(f"RUN_PN_COUNT       : {RUN_PN_COUNT}")
    print(f"RUN_CASES_PER_PN   : {RUN_CASES_PER_PN}")
    print(f"RUN_FREQ_MAX_GHZ   : {RUN_FREQ_MAX_GHZ}")
    print(f"RUN_TDR_POINTS     : {RUN_TDR_POINTS}")
    print(f"IMPEDANCE TOL      : +/-{IMPEDANCE_TOLERANCE_PCT*100:.1f}%")
    generate_dataset(
        output_dir=RUN_OUTPUT_DIR,
        pn_count=RUN_PN_COUNT,
        cases_per_pn=RUN_CASES_PER_PN,
        freq_points=RUN_FREQ_POINTS,
        tdr_points=RUN_TDR_POINTS,
        freq_max_ghz=RUN_FREQ_MAX_GHZ,
        seed=RUN_SEED,
        save_features=RUN_SAVE_FEATURES,
    )


if __name__ == "__main__":
    main()
