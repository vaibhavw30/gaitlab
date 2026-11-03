---
description: Record an architecture decision
argument-hint: <short title>
---

Create a new ADR titled "$ARGUMENTS".

1. Find the highest number in `docs/adr/` and use the next one (4 digits).
2. Copy `docs/adr/TEMPLATE.md` to `docs/adr/NNNN-<kebab-title>.md` and fill it in from the current conversation and code: Context (cite result files or papers), Decision, Consequences. Status `accepted` unless I said otherwise; today's date.
3. If it supersedes an ADR, update that ADR's status line.
4. If it changes a contract, update `docs/TECHNICAL_PLAN.md` in the same change.
