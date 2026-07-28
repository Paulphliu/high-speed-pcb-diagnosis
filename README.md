# High-Speed PCB Diagnosis

A simulation-driven and AI-assisted framework for high-speed PCB signal-integrity screening using VNA, TDR, and Delta-L features.

## Overview

High-speed PCB signal-integrity diagnosis normally requires experienced engineers to interpret frequency-domain and time-domain measurement results.

This project explores an automated diagnostic framework that combines:

- VNA S-parameter features
- TDR impedance features
- Delta-L insertion-loss features
- Machine-learning-based defect classification
- Measurement-level and lot-level diagnosis

The objective is to provide a reproducible research platform for preliminary screening of high-speed PCB signal-integrity abnormalities.

## Diagnostic Workflow

The general workflow is:

1. Generate electrically meaningful PCB signal-integrity cases.
2. Simulate VNA, TDR, and Delta-L responses.
3. Extract diagnostic features from the simulated responses.
4. Train a machine-learning classification model.
5. Perform measurement-level defect screening.
6. Aggregate multiple measurements for lot-level diagnosis.
7. Report primary and secondary engineering interpretations.

## Diagnostic Conditions

The framework is designed to study conditions including:

- Normal transmission line
- Impedance shift
- Excess insertion loss
- Conductor roughness variation
- Dielectric-property variation
- Local trace discontinuity
- Crosstalk
- Via discontinuity
- Via-stub-related anomaly

The supported labels may be refined as the project develops.

## Simulation-Derived Dataset

All example datasets in this repository are generated from electrical models and controlled parameter variations.

They are based on engineering concepts including:

- Transmission-line theory
- Characteristic-impedance variation
- Frequency-dependent conductor loss
- Dielectric loss
- Surface-roughness effects
- Via and stub discontinuities
- Localized geometrical defects
- Coupling and crosstalk effects

Unless explicitly stated otherwise, the data provided in this repository are not production measurement data.

## Project Scope

This repository is intended for:

- Research reproducibility
- Engineering education
- Algorithm evaluation
- Preliminary signal-integrity screening
- Future comparison with measured VNA and TDR data

The framework is not intended to replace calibrated laboratory measurements, full-wave electromagnetic simulation, or professional signal-integrity analysis.

## Repository Contents

The repository will include:

```text
high-speed-pcb-diagnosis/
├── src/              # Python source code
├── sample_data/      # Example input datasets
├── models/           # Trained model files
├── examples/         # Demonstration scripts
├── results/          # Example diagnostic outputs
├── docs/             # Technical documentation
├── requirements.txt  # Python dependencies
└── README.md
