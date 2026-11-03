# ADR-0003: Split and bootstrap by session

- Status: accepted
- Date: 2026-11-03

## Context
Windows from the same session share lighting, shoes, speed and fatigue state and are near-duplicates. A random window split leaks.

## Decision
`data/splits.json` assigns whole sessions: the last 2 by date go to test, the previous one to dev, the rest to train. It is frozen once written. Baselines use leave-one-session-out CV on train + dev. Bootstrap CIs resample sessions, then windows.

## Consequences
CIs will be wide with few sessions. That is the honest answer, and it is reported as is.
