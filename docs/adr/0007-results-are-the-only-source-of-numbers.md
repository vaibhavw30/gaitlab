# ADR-0007: results/*.json is the only source of reported numbers

- Status: accepted
- Date: 2026-11-03

## Context
Placeholder numbers have leaked into résumé drafts before. A fabricated number is the one error that ends an interview process.

## Decision
Only `report.results.write_result()` writes results, with provenance (git SHA, dirty flag, config hash, data manifest hash, hardware, versions, synthetic flag). The README results block is generated. `scripts/check_readme_numbers.py` runs in CI.

## Consequences
Every claim is traceable to a file and a commit. Updating a number means rerunning the script, not editing text.
