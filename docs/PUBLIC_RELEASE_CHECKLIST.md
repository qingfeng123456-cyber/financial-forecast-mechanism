# Public release checklist

- Replace the contributor entity in `CITATION.cff` with the final author list.
- Confirm that a public GitHub account is compatible with the journal's
  double-anonymous review procedure; otherwise release after review or use an
  editor-approved anonymous archive.
- Do not commit `.npz` market panels, model weights, raw logs, or returned ZIP
  archives.
- Run `python scripts/static_validate.py` and resolve every reported failure.
- Confirm that manuscript funding, conflict, author-contribution, and data
  statements match the final submission.
- Create a tagged release and attach the generated source archive only after
  reviewing `RELEASE_MANIFEST.json`.
