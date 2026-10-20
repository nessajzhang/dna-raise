#!/usr/bin/env python3
"""Rebuild the public-endpoint and full assurance grids from inherited CSV data."""
from pathlib import Path

import inherited_submission_figures as source

root = Path(__file__).resolve().parents[1]
source.DATA = root / "source_data/inherited"
source.FIGURES = root / "figures"
source.setup_style()
source.paper4_measurement_layers()
source.paper4_policy_assurance()
print("Rebuilt Figure S1 and full four-panel assurance grid from inherited CSV")
