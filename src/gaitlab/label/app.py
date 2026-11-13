"""Gradio labeling app (TECHNICAL_PLAN §6, US-5, FR-7). TASKS M2.

    gaitlab label                      normal mode on data/processed/*/windows.parquet
    gaitlab label --blind 40           blind re-grade of 40 labeled windows (seeded)
    gaitlab label --synthetic          the same on gaitlab synth output (labels stay there)

Layout: left, the window's overlay clip ([t_start, t_end] of its clip, skeleton + IC/TO markers,
on the source video when it's found, else on a plain canvas); right, the feature table (median,
IQR), a 1-5 grade, tag checkboxes (configs/features.yaml vocabulary), a cue box and
Back / Skip / Save. Keys: 1-5 set the grade, Enter saves (not while typing in the cue box).

- Normal mode: windows without a normal-mode label (``label.queue.normal_queue``).
- Blind mode: ``label.queue.blind_sample``; no window id, no feature table, and the overlay has
  no angle / cadence text. Rows are written with ``mode="blind_regrade"``.
- Every save is one ``data.sessions.append_label`` call: validated, locked, append-only.
- Real labels go to ``<root>/labels/labels.csv``. Synthetic windows can be labeled only with
  ``--synthetic``, into ``data/synthetic/labels/hand_labels.csv`` (never synth's rule-label
  ``labels.csv``; CLAUDE.md rule 3).
- Overlay clips are rendered on first view into ``<root>/labels/overlays/`` (under ``data/``,
  git-ignored) and reused.

gradio is imported only inside ``build_app`` and ``launch`` (``app`` extra).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from gaitlab.data.sessions import append_label, load_labels
from gaitlab.label.queue import LabelSession, blind_sample, check_synthetic, normal_queue

KEYS_JS = """
<script>
document.addEventListener("keydown", (e) => {
  const t = e.target;
  const typing = t && (t.tagName === "TEXTAREA" || (t.tagName === "INPUT" && t.type === "text"));
  if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
  if ("12345".includes(e.key) && e.key.length === 1) {
    const r = document.querySelector(`#grade input[type=radio][value="${e.key}"]`);
    if (r) { r.click(); e.preventDefault(); }
  } else if (e.key === "Enter") {
    const b = document.querySelector("#save");
    if (b) { b.click(); e.preventDefault(); }
  }
});
</script>
"""


@dataclass
class LabelContext:
    """Everything the UI needs: the session, where labels go, and how to get a window's clip."""

    root: Path
    labels_csv: Path
    session: LabelSession
    feature_order: list[str]
    vocab: list[str]
    overlay: Callable[[str], Path]


class OverlayCache:
    """Renders a window's overlay clip once, into ``<root>/labels/overlays/<session>/``."""

    def __init__(
        self,
        root: Path,
        windows: Mapping[str, Mapping[str, Any]],
        backend: str,
        pipeline_cfg,
        numbers: bool,
    ) -> None:
        self.root, self.windows, self.backend = root, windows, backend
        self.cfg, self.numbers = pipeline_cfg, numbers
        self._clips: dict[tuple[str, str], tuple[Any, Any]] = {}
        self._videos = _session_videos(root)

    def _clip(self, sid: str, cid: str):
        from gaitlab.gait.features import analyze_clip
        from gaitlab.types import KeypointSeq

        if (sid, cid) not in self._clips:
            npz = self.root / "interim" / sid / f"{cid}.{self.backend}.npz"
            if not npz.exists():
                raise FileNotFoundError(f"{npz} not found: run `gaitlab extract` for {sid}")
            seq = KeypointSeq.load(str(npz))
            self._clips[(sid, cid)] = (seq, analyze_clip(seq, self.cfg))
        return self._clips[(sid, cid)]

    def __call__(self, window_id: str) -> Path:
        from gaitlab.viz.overlay import render_overlay

        w = self.windows[window_id]
        sid, cid = str(w["session_id"]), str(w["clip_id"])
        start = w.get("start_stride", window_id.rsplit(":", 1)[-1])
        kind = "full" if self.numbers else "blind"
        out = self.root / "labels" / "overlays" / sid / f"{cid}.{start}.{self.backend}.{kind}.mp4"
        if out.exists():
            return out
        seq, analysis = self._clip(sid, cid)
        video = self._videos.get((sid, cid)) or _raw_video(self.root, sid, cid)
        return render_overlay(
            seq, analysis, str(out), video_path=str(video) if video else None,
            t_range=(float(w["t_start"]), float(w["t_end"])), numbers=self.numbers,
        )  # fmt: skip


def _session_videos(root: Path) -> dict[tuple[str, str], Path]:
    """(session, clip) → source video from ``<root>/sessions.csv`` rows that point at one."""
    path = root / "sessions.csv"
    if not path.exists():
        return {}
    from gaitlab.data.sessions import load_sessions

    out = {}
    for r in load_sessions(path):
        p = Path(r.path)
        p = p if p.is_absolute() or p.exists() else root / p
        if p.suffix != ".npz" and p.exists():
            out[(r.session_id, r.clip_id)] = p
    return out


def _raw_video(root: Path, sid: str, cid: str) -> Path | None:
    p = root / "raw" / sid / f"{cid}.mp4"
    return p if p.exists() else None


def load_windows(root: Path) -> list[dict[str, Any]]:
    """Every row of ``<root>/processed/*/windows.parquet``."""
    import pandas as pd

    files = sorted((root / "processed").glob("*/windows.parquet"))
    if not files:
        return []
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).to_dict("records")


def prepare(
    root: str | Path,
    *,
    synthetic: bool,
    backend: str,
    rater: str = "vw",
    blind: int = 0,
    seed: int = 0,
    labels_csv: str | Path | None = None,
    pipeline_config: str = "configs/default.yaml",
    features_config: str = "configs/features.yaml",
) -> LabelContext:
    """Load windows and labels, pick the queue, and set up the writer and overlay renderer."""
    from gaitlab.config import load_features_config, load_pipeline_config

    root = Path(root)
    windows = load_windows(root)
    check_synthetic({bool(w.get("synthetic", False)) for w in windows}, synthetic)
    by_id = {str(w["window_id"]): w for w in windows}
    out = Path(labels_csv) if labels_csv else root / "labels" / (
        "hand_labels.csv" if synthetic else "labels.csv"
    )  # fmt: skip
    labels = load_labels(out) if out.exists() else []
    if blind:
        if not any(r.mode == "normal" for r in labels):
            raise ValueError(f"nothing to re-grade: {out} has no normal-mode labels yet")
        ids = [w for w in blind_sample(labels, blind, seed) if w in by_id]
    else:
        ids = normal_queue(windows, labels)
    fcfg = load_features_config(features_config)
    vocab = list(fcfg.tags)
    mode = "blind_regrade" if blind else "normal"
    session = LabelSession(ids, by_id, mode=mode, rater=rater, vocab=vocab,
                           write=partial(append_label, out))  # fmt: skip
    overlay = OverlayCache(root, by_id, backend, load_pipeline_config(pipeline_config),
                           numbers=not blind)  # fmt: skip
    return LabelContext(root, out, session, list(fcfg.feature_order), vocab, overlay)


def build_app(ctx: LabelContext):
    """The Gradio Blocks UI around ``ctx.session`` (not launched)."""
    import gradio as gr

    s = ctx.session

    def view(status: str = ""):
        wid = s.current()
        video = None
        if wid is not None:
            try:
                video = str(ctx.overlay(wid))
            except (FileNotFoundError, ValueError) as e:
                status = f"{status}\n\nNo clip for this window: {e}".strip()
        return (f"### {s.title()}", video, s.features(ctx.feature_order), None, [], "",
                f"{s.progress()}  \n{status}".strip())  # fmt: skip

    def on_save(grade, tags, cue):
        try:
            row = s.save(grade, tags or [], cue or "")
        except ValueError as e:
            title, video, table, _, _, _, _ = view()
            return title, video, table, grade, tags, cue, f"{s.progress()}  \n**Not saved:** {e}"
        return view(f"Saved {row.window_id if not s.blind else 'grade'} = {row.grade}")

    def on_skip():
        s.skip()
        return view("Skipped (nothing written)")

    def on_back():
        s.back()
        return view("Back: saving again adds a new row; the later one wins")

    with gr.Blocks(title="GaitLab labeling") as demo:
        title = gr.Markdown()
        with gr.Row():
            with gr.Column(scale=3):
                video = gr.Video(label="window", interactive=False, autoplay=True, loop=True)
            with gr.Column(scale=2):
                table = gr.Dataframe(headers=["feature", "median", "IQR"], interactive=False,
                                     visible=not s.blind, label="features")  # fmt: skip
                grade = gr.Radio(["1", "2", "3", "4", "5"], label="grade (keys 1-5)",
                                 elem_id="grade")  # fmt: skip
                tags = gr.CheckboxGroup(ctx.vocab, label="tags")
                cue = gr.Textbox(label="cue (one line)", lines=2)
                with gr.Row():
                    back = gr.Button("Back")
                    skip = gr.Button("Skip")
                    save = gr.Button("Save (Enter)", variant="primary", elem_id="save")
                status = gr.Markdown()
        outputs = [title, video, table, grade, tags, cue, status]
        save.click(on_save, inputs=[grade, tags, cue], outputs=outputs)
        skip.click(on_skip, outputs=outputs)
        back.click(on_back, outputs=outputs)
        demo.load(view, outputs=outputs)
    return demo


def launch(ctx: LabelContext, port: int = 7860, open_browser: bool = True) -> None:
    """Serve the app on localhost only (labels and video never leave the machine)."""
    demo = build_app(ctx)
    demo.launch(server_name="127.0.0.1", server_port=port, inbrowser=open_browser,
                allowed_paths=[str(ctx.root)], head=KEYS_JS)  # fmt: skip
