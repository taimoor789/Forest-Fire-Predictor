"""
ML rebuild package -- separate from the live serving path (fire_risk.py,
main.py, collect_weather_grid_eccc.py). Nothing here runs in production
until Stage 12. Rebuilds the training pipeline after a re-audit found the
original model_components/*.pkl was built on leaked labels and formulas that
had silently diverged from production; see docs/DATA_PROVENANCE.md.

Every stage imports fire_risk.py's actual FWI methods rather than
reimplementing them, so that divergence can't happen again.
"""
