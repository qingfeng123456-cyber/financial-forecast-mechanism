# Reproducibility guide

## Two evidence layers

The checked-in `evidence/` directory contains compact summaries from the
completed runs. Re-execution creates a new run and must not overwrite those
historical summaries. The full private archives remain outside this GitHub
release because they contain model weights, forecast arrays, raw logs, and many
large diagnostic files.

## Re-execution

Each package is independent. Enter one directory under `model_code/`, install its
listed dependencies manually, provide the exact three data panels, and run
`python main.py`. The configurations do not perform dependency installation,
automatic sweeps, retries, or wall-clock termination.

## Interpretation

Use dates as the inferential unit. Keep task-specific scores separate. A
positive mechanism diagnostic is descriptive and does not establish forecast
improvement. The retrospective strongest comparator is distinct from a
validation-selected comparator, and intervals conditional on retrospective
selection do not include selection uncertainty.

## Integrity checks

`RELEASE_MANIFEST.json` records every released file and SHA-256 digest. Run the
static validator before creating a GitHub release. Runtime claims require an
actual training-host execution and must not be inferred from static checks.
