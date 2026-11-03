---
description: Implement the next open tasks of a milestone from docs/TASKS.md
argument-hint: M0 | M1 | M2 | M3 | M4 | M5
---

Work on milestone $ARGUMENTS.

1. Read `docs/TASKS.md` and list the unchecked (`[ ]`) and scaffolded (`[~]`) items under $ARGUMENTS. Skip items marked "(user task)"; list them at the end for the user instead.
2. For the first item, read the relevant section of `docs/TECHNICAL_PLAN.md` and any ADR it references.
3. Write or extend the test first. Show me the failing test output.
4. Implement until `uv run pytest -m "not gpu and not data"` and `uv run ruff check .` are clean.
5. If the item produces a result file, write it only through `gaitlab.report.results.write_result()`, with `synthetic` set correctly.
6. Tick the box in `docs/TASKS.md`. If you made a design choice the plan didn't specify, add an ADR from `docs/adr/TEMPLATE.md`.
7. Stop and summarize: what changed, test status, and the next item. Don't start the next item unless I say so.

Never write a metric value into any doc or docstring; see CLAUDE.md hard rule 1.
