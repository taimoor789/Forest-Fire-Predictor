# shadow-eval-log

Accumulating data branch for the live ML shadow-mode observation period —
see `docs/PREREGISTRATION.md`'s 2026-09-14 amendment on `main` for the full
design and the pre-committed tripwire conditions.

**This branch is append-only and is never force-pushed**, unlike `deploy`
(which is deliberately rebuilt as a single orphan commit daily to stay
small). Two isolated GitHub Actions workflows write to disjoint subpaths
here:

- `hotspots/` — CWFIS satellite fire-detection pulls (`ml/cwfis_hotspots.py`,
  `.github/workflows/shadow-eval-hotspots.yml`), 3x/day. The source is a
  rolling 24h window with no backfill, so this data is unrecoverable if a
  pull is ever missed.
- `predictions/` — daily FWI/ML tier snapshots (`ml/shadow_snapshot.py`,
  `.github/workflows/shadow-eval-snapshot.yml`), triggered after each
  successful `daily-pipeline.yml` run.
- `grid_cells.csv` — the in-domain cell set, derived from `fwi_predictions.json`
  once and asserted-consistent on every later snapshot (`data/grid_domain_v1.parquet`
  is not tracked in git and unavailable in CI, so this is the CI-safe
  substitute).

Intended lifetime: through the observation period only (originally ~1 week
from 2026-09-14, extendable per the pre-registered minimum-positives rule,
hard stop 2026-10-15). Once the tripwire decision is recorded as a dated
`docs/PREREGISTRATION.md` amendment on `main`, this branch's job is done —
safe to archive or delete after that, the decision itself lives on `main`.

Read with `ml/shadow_report.py` (run manually from `main`, not scheduled).
