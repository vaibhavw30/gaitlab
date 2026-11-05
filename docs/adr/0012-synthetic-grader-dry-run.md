# ADR-0012: Grader dry run on synthetic data: split, prediction files, result layout

- Status: accepted
- Date: 2026-11-04

## Context
Baselines, SFT export and grader evaluation should run end to end before real labels exist, on `gaitlab synth` output (rule-generated labels, so everything is synthetic: CLAUDE.md rule 3). TECHNICAL_PLAN §7.1 and EVALUATION G1/G3/G5 fix the models and metrics, but not where the synthetic split lives, the file handed from models to `eval`, how ties in the LightGBM grid break, or the shape of the result JSON.

## Decision
1. **Synthetic split:** `make_session_split` over `data/synthetic/sessions.json` (by date: last 2 sessions test, 1 dev, rest train), written once to `data/synthetic/splits.json` with `write_split_frozen`. It is separate from `data/splits.json`, which `--synthetic` never touches. `gaitlab synth` deletes the synthetic split, since it regenerates the sessions it was derived from.
2. **Real data is not wired yet:** `baselines`, `build-sft` and `eval` without `--synthetic` exit with a message until real labels and `data/splits.json` exist (M2). The model and eval code is the same for both.
3. **Prediction files** are the hand-off to `eval`: `<dir>/<model>.csv` with `window_id, pred_grade, p1..p5, tags, cue, synthetic` (blank `p*` when a model has no probabilities, as for rules), plus optional `<model>.meta.json` (e.g. the chosen LightGBM grid point and its per-point LOSO QWK), copied into the ladder as `selection`. Every model must cover exactly the same test windows, or `eval` fails.
4. **Baselines:** LR (median imputation, standardization, lbfgs multinomial, C = 1) and LightGBM (grid `num_leaves ∈ {7, 15} × n_estimators ∈ {100, 300}`, ties go to the earlier, smaller grid point) are fit on train + dev, with the grid chosen by leave-one-session-out QWK there. A training set with one grade gives a model that predicts that grade with P = 1. Tags come from one-vs-rest LR per tag, shared by LR and LightGBM; a tag that is constant in training is predicted constant.
5. **Result layout:** `ladder.json` metrics = `{split, n_windows, n_sessions, order, models: {name: grade_metrics + selection}, best: {model, qwk}}`, ordered by the EVALUATION §2 ladder. `tags.json` = macro-F1 / per-tag F1 per model. `calibration.json` = ECE, Brier and a 10-bin reliability table per model with probabilities, plus `no_probabilities`. `eval` refuses `synthetic=False` when any label (rater `synthetic`) or prediction is synthetic.
6. **Plots** read only those JSON files and processed `strides.parquet` (new `condition` and 101-point `knee_flex_cycle_deg` columns, so the knee-cycle chart doesn't re-run pose analysis). Charts from synthetic inputs are titled and watermarked SYNTHETIC and refuse to write under `docs/`.

## Consequences
- With rule labels, the rules baseline reproduces its own labels on the synthetic test split. That checks the plumbing and says nothing about grading quality.
- Two test sessions make the session-bootstrap CIs coarse; that is a property of the dry run, not of the metric code.
- CI installs `ml` and `viz`, so the chart-rendering tests run there too (CI gained `--extra viz` after this ADR).
