# GaitLab on DeltaAI (GH200, aarch64): runbook

Project: `~/Downloads/gaitlab` on the laptop, `~/gaitlab` on the cluster. Every command is marked
**LAPTOP** or **CLUSTER** (login node, or inside a job). The session runs only the committed scripts
in `scripts/cluster/`. Prof. Chou approved up to **15 GPU-hours** on allocation `bhhv-dtai-gh`, for
this project only.

**How sure each part is.**
- **[checked]**: from GaitLab's own files (`pyproject.toml`, `uv.lock`, `CLAUDE.md`, `docs/`).
- **[docs]**: from NCSA's public DeltaAI documentation, read 2026-11-04, with the page linked.
  `B` = https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/
- **[unverified]**: not stated in either. Round 1 (login node) or the `rf_gate` job checks it.

---

## What DeltaAI is (from NCSA's docs)

| Topic | What the docs say | Source |
|---|---|---|
| Login OS | Login banner in an example: "OS: SLES 15.5 ... CrayOS NVIDIA Grace-Grace aarch64" [docs]. **[checked] round 1 (2026-11-05): the login node runs SLES 15 SP6, aarch64, glibc 2.38.** The compute-node OS and glibc are [unverified] until `rf_gate`. | [B/user-guide/vscode/code-server.html](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/vscode/code-server.html) |
| GPU nodes | 4x GH200 per node; "A full node has 4 GPUs, 288 CPU cores, and ~440 GB of available RAM, so 1/4 node = 1 GPU ≈ 72 CPU cores ≈ 110 GB of RAM." Login nodes have no GPUs. | [B/user-guide/running-jobs.html](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/running-jobs.html), [B/user-guide/architecture.html](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/architecture.html) |
| Driver / max CUDA | **Not stated** [unverified]. The only driver figure is old NGC 24.09 output on the containers page ("CUDA Forward Compatibility mode ENABLED ... kernel driver version 535.183.06"). The site PyTorch modules 2.11.0 and 2.12.0 ship CUDA 13.0 wheels and load a site `cuda/13.1.1` module, which suggests a newer driver now. The `cudatoolkit` module is CUDA 12.9. | [B/user-guide/containers.html](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/containers.html), [B/user-guide/python/pytorch.html](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/python/pytorch.html), [B/user-guide/prog-env.html](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/prog-env.html) |
| Storage | `/u/$USER` (HOME): 100 GB, 750,000 files, 30-day snapshots. `/projects`: 500 GB by default. `/work/nvme` and `/work/hdd`: 1000 GB by default. None of them is purged. "DeltaAI does not provide a /scratch file system." Node `/tmp` is purged after each job. "do not put symbolic links from $HOME to the /projects and /work spaces." `quota` shows usage. | [B/user-guide/data-mgmt.html](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/data-mgmt.html) |
| venv location | `/work/nvme` is the "Recommended default" for Python environments. | [B/user-guide/software.html](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/software.html) |
| Backups | "You are responsible for backing up your files in /projects and /work." | [B/user-guide/architecture.html](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/architecture.html) |
| Containers | "Container support on DeltaAI is provided by Apptainer". Prebuilt NGC images live in `/sw/user/NGC_containers/`, and "The most recent container is pytorch_26.01-py3.sif". `--nv` is used in every example. **[checked] round 1: `apptainer version 1.4.2-111.1` on the login node; the directory also holds `pytorch_26.07-py3.sif`, newer than the docs mention.** Whether pulls are allowed on login nodes is not stated [unverified]. | [B/user-guide/containers.html](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/containers.html), [B/user-guide/python/pytorch.html](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/python/pytorch.html) |
| Internet | Not stated in the docs. **[checked] round 1: the login node reaches pypi.org, pypi.nvidia.com and huggingface.co (HTTP 200).** Compute nodes are [unverified]; we assume none, jobs run offline, and `rf_gate` reports it. | — |
| Charging | "The charge unit for DeltaAI is the service unit (SU). This corresponds to the equivalent use of one Grace Hopper GH200 for 1 hour." "Charges are based on the number of GH200 GPUs needed to fullfill the requested memory or CPU cores, whichever is larger." Charge factor: ghx4 1.0, **ghx4-interactive 2.0**. The balance is shown by `accounts`. | [B/user-guide/job-accounting.html](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/job-accounting.html), [B/user-guide/running-jobs.html](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/running-jobs.html) |
| Partitions | ghx4: 48 h max. ghx4-interactive: 2 h max, one job at a time. Default wall time is 30 min. Nodes are shared by default. `--nice` isn't mentioned (it's standard Slurm). **[checked] 2026-11-05: `--nice=1000` outweighs the whole priority (age + fairshare + partition), so `sprio` showed priority 1 behind ~2100 pending ghx4 jobs. `scontrol update jobid=<id> nice=0` is allowed for users and restored it; jobs now use `--nice=0`.** | [B/user-guide/running-jobs.html](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/running-jobs.html) |
| Login nodes | "Do not run production jobs on the login nodes (very short time debug tests are fine)." Use scp/rsync "for small to modest transfers". | [B/user-guide/citizenship.html](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/citizenship.html), [B/user-guide/data-mgmt.html](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/data-mgmt.html) |

**What that means for the budget.**
- Every job requests 1 GPU, 72 cores and 100 GB: the most a 1-GPU job can take and still be billed as 1 GH200 (72 cores and ~110 GB per GH200). Charged hours equal GPU-hours on ghx4, and nothing is held back to save budget.
- An interactive session is billed double, so we use short `sbatch` jobs only.
- `scripts/cluster/budget.sh` applies these rules to `sacct` and warns at 14.0 of the 15.0. The 15.0 is the budget, not a hard cap: past 14.0, `submit.sh` asks before submitting.

## What the project needs (from its files)

| | | |
|---|---|---|
| Python | 3.11 through `uv` (`env.sh` sets `UV_PYTHON=3.11`) | [checked] `requires-python = ">=3.11,<3.13"` |
| Locked versions | torch 2.14.1, torch-tensorrt 2.14.0, tensorrt-cu13 11.1.0.106, transformers 5.22.1, trl 1.17.1, peft 0.21.2 | [checked] `uv.lock` |
| torch on Linux | a CUDA 13.0 build: it depends on `cuda-toolkit` 13.0, `nvidia-cudnn-cu13`, `nvidia-nccl-cu13` | [checked] `uv.lock`. So **torch itself** needs a CUDA 13 driver, not only TensorRT. |
| glibc on aarch64 | `llm` + `dev` wheels need at most `manylinux_2_28`. The TensorRT wheels (`tensorrt-cu13-bindings`, `-libs`) are `manylinux_2_35` only. | [checked] `uv.lock`. Below glibc 2.35, `--extra trt` won't install; see "Fallback". The login node has 2.38 [checked]. |
| Models | `google/gemma-3-1b-it`, `google/gemma-3-270m-it`; gated (accept the Gemma license) | [checked] |
| Model cache | `$HF_HOME` (set by `env.sh`), otherwise `<repo>/.hf_cache` | [checked] `llm/runtime.py` |
| Precision | bf16 on CUDA; eager attention for training, SDPA for inference | [checked] |

**Three things that decide whether the uv environment works on DeltaAI:**
1. **[unverified]** The driver supports CUDA 13 (needed by torch and TensorRT). `rf_gate` checks this.
2. **[checked on the login node: glibc 2.38; compute node unverified]** glibc ≥ 2.35 for TensorRT. `rf_gate` checks the compute node. If it fails, LoRA work still runs in the uv env and only B2 needs the container.
3. **[checked]** Use the project's uv environment, not `python/miniforge3_pytorch`. torch-tensorrt 2.14.0 needs torch 2.14.x, and the newest module ships 2.12.0 [docs].

---

## The scripts (`scripts/cluster/`)

| Script | Where | Does |
|---|---|---|
| `laptop.sh` | LAPTOP (sourced) | host and ssh options, shared by the others |
| `ssh.sh [script]` | LAPTOP | ssh with the shared master connection; with a script, runs it on the login node |
| `probe.sh` | login node (via `ssh.sh`) | round 1, read-only: host, OS, glibc, quota, allocation dirs, apptainer, internet |
| `rsync-up.sh [--dry-run]` | LAPTOP | sends code, configs, `.git`, `data/synthetic`; refuses if anything else from `data/`, caches, runs, results or `.env` would go |
| `env.sh [job]` | CLUSTER (sourced) | cache, venv and HF paths under `GAITLAB_STORE`; `job` mode adds `UV_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` |
| `setup-login.sh uv\|sync\|models` | login node | install uv, build the venv from `uv.lock`, download Gemma |
| `gate.py` | in `rf_gate` | driver/CUDA 13, torch CUDA build on a GH200, TensorRT imports → `GATE RESULT PASS\|FAIL` |
| `rf_gate.sbatch` (15 min), `rf_gputest.sbatch` (45 min), `rf_synth.sbatch` (45 min) | jobs | ghx4, 1 GPU, 72 cores, 100 GB (billed as 1 GH200), `--nice=0`, logs to `logs/<name>_<jobid>.out` |
| `budget.sh [--next HH:MM:SS [partition]]` | login node | charged GH200-hours of `rf_*` jobs since 2026-11-04, used/remaining of 15.0; exit 3 (ASK) past 14.0 |
| `submit.sh rf_…` | login node | checks the job header, runs `budget.sh --next <its limit>`, then `sbatch` |
| `rsync-down.sh <run_id> [--with-results]` | LAPTOP | pulls `adapter/`, `trainer_state.json`, `run.json`, `config.json`, `SYNTHETIC`, `logs/rf_*.out` and, with the flag, `results/synthetic/**/*.json` |

**One Duo prompt per ~4 h.** The first ssh or rsync opens a master connection, and later ones reuse it for 4 h.
- ssh options: `-o ControlMaster=auto -o ControlPath=~/.ssh/dtai-%C -o ControlPersist=4h`
- rsync gets the same through `-e "ssh -o ControlMaster=auto -o ControlPath=~/.ssh/dtai-%C -o ControlPersist=4h"`, so either one can open the master.
- Close it at the end with `scripts/cluster/ssh.sh -O exit`.
- If NCSA refuses multiplexing [unverified], every command just asks for Duo again.

---

## Session: steps 1–7

### 1. LAPTOP: round 1 (read-only), then send the repo

```bash
cd ~/Downloads/gaitlab && scripts/cluster/ssh.sh scripts/cluster/probe.sh
```

The output must say `OK: DeltaAI login node`; `WRONG CLUSTER` means stop. Use the `quota` and `ls -ld /work/nvme/bhhv*` output to pick `GAITLAB_STORE`, set it at the top of `scripts/cluster/env.sh`, and commit. (Done 2026-11-05: `/work/nvme/bhhv/vwudaru/gaitlab`, because `$HOME` was mostly full.) Then:

```bash
cd ~/Downloads/gaitlab && scripts/cluster/rsync-up.sh --dry-run
cd ~/Downloads/gaitlab && scripts/cluster/rsync-up.sh
```

### 2. LAPTOP: log in

```bash
cd ~/Downloads/gaitlab && scripts/cluster/ssh.sh
```

The same as `ssh -o ControlMaster=auto -o ControlPath=~/.ssh/dtai-%C -o ControlPersist=4h vwudaru@dtai-login.delta.ncsa.illinois.edu`. The prompt must show `gh-loginNN`; `dt-loginNN` is Delta, the wrong cluster.

### 3. CLUSTER (login node): uv, the environment, the models, once

```bash
cd ~/gaitlab && scripts/cluster/setup-login.sh uv && scripts/cluster/setup-login.sh sync
```

`sync` builds `llm`+`dev` first, then tries `llm`+`trt`+`dev`. A `SYNC trt: FAILED` line leaves a working LoRA environment; paste the error. On the login node `torch.version.cuda` must print 13.x. `torch.cuda.is_available()` would be False there, which is expected because there's no GPU.

Hugging Face login. It's interactive: paste your Read token at the prompt yourself, never into a file or chat. The token is stored in `$HF_HOME/token` under `GAITLAB_STORE`.

```bash
cd ~/gaitlab && source scripts/cluster/env.sh && uv run hf auth login
```

```bash
cd ~/gaitlab && scripts/cluster/setup-login.sh models
```

### 4. CLUSTER: the gate, then the GPU smoke tests (batch jobs)

```bash
cd ~/gaitlab && scripts/cluster/submit.sh rf_gate
```

When it has left `squeue --me`: `cat logs/rf_gate_<jobid>.out`. **Nothing else runs on a GPU unless it ends in `GATE RESULT PASS`.**

| Gate line | Pass means | If it fails |
|---|---|---|
| `driver_supports_cuda13` | `nvidia-smi` shows CUDA Version 13.x | the driver is too old for the lock: use "Fallback", or a CUDA 12 torch index (a lock change; ask first) |
| `torch_cuda_build`, `torch_cuda_available`, `device_is_gh200`, `cuda_bf16_matmul` | the locked aarch64 torch wheel is a CUDA build and runs on the GH200 | see the error; usually the driver |
| `import_tensorrt`, `import_torch_tensorrt` | B2 can run in the uv env | usually glibc < 2.35 (no TensorRT wheels): B2 moves to "Fallback"; the LoRA steps still run |

```bash
cd ~/gaitlab && scripts/cluster/submit.sh rf_gputest
```

This is `pytest -m gpu`: the 4-step bf16 LoRA on 270M, CUDA-vs-CPU grade argmax, `cuda_time`, and the TensorRT compile of the 270M scoring pass (skipped if TensorRT isn't installed). If it passes on a GH200, tick TASKS M3 "GPU smoke test on the M3 host" on the laptop.

### 5. CLUSTER: 1B LoRA dry run on synthetic data

```bash
cd ~/gaitlab && scripts/cluster/submit.sh rf_synth
```

Then `cat logs/rf_synth_<jobid>.out`. Its last line is the run directory `runs/1b-<UTC time>-<hash>`. The output is synthetic and is never reported (rule 3).

### 6. CLUSTER: GPU-hours (before every job)

```bash
cd ~/gaitlab && scripts/cluster/budget.sh
```

`submit.sh` already runs `budget.sh --next <the job's time limit>` and asks `[y/N]` at exit 3. Paste its output every time; the tally in each reply comes from it.

### 7. LAPTOP: bring back the run and the logs

```bash
cd ~/Downloads/gaitlab && scripts/cluster/rsync-down.sh <run_id from step 5> && scripts/cluster/ssh.sh -O exit
```

It brings back only `runs/<run_id>/{adapter/, trainer_state.json, run.json, config.json, SYNTHETIC}` and `logs/rf_*.out`. Both `runs/` and `logs/` are git-ignored.

---

## Fallback: NGC PyTorch container (plan only, not tried)

Use this if the locked wheels can't install (TensorRT on glibc < 2.35) or can't see the driver. NGC PyTorch images bundle torch, TensorRT and Torch-TensorRT built for each other, on a newer base OS. **`pyproject.toml` and `uv.lock` stay unchanged.** The repo goes into a venv inside the container that uses the container's torch, and torch is never re-resolved.

1. **Login node: pick the image.** DeltaAI already provides `/sw/user/NGC_containers/pytorch_26.01-py3.sif` [docs]. Its torch, TensorRT and Torch-TensorRT versions are [unverified]; print them. Only if a newer one is needed, pull it into the store; whether pulls are allowed on login nodes is [unverified].

   ```bash
   cd ~/gaitlab && source scripts/cluster/env.sh && SIF=/sw/user/NGC_containers/pytorch_26.01-py3.sif && apptainer exec "$SIF" python -c "import sys, torch, tensorrt, torch_tensorrt; print(sys.version.split()[0], torch.__version__, torch.version.cuda, tensorrt.__version__, torch_tensorrt.__version__)"
   ```
2. **Login node: venv inside the container.** Use `--system-site-packages`, so the container's torch, TensorRT, Torch-TensorRT, numpy and CUDA libraries are the ones imported.

   ```bash
   cd ~/gaitlab && source scripts/cluster/env.sh && SIF=/sw/user/NGC_containers/pytorch_26.01-py3.sif && apptainer exec "$SIF" python -m venv --system-site-packages "$GAITLAB_STORE/venv-ngc"
   ```
3. **Login node: install the rest from the lock, without dependency resolution.**
   - Export the locked `llm` + `dev` pins and drop everything the container provides: torch, torch-tensorrt, tensorrt*, triton, nvidia-*, cuda-*, numpy.
   - Install with `--no-deps`, then the repo itself with `--no-deps -e .`.
   - `pip check` then lists any pin the container's torch doesn't satisfy (most likely transformers/peft/trl). That's the point to stop and decide.

   ```bash
   cd ~/gaitlab && source scripts/cluster/env.sh && SIF=/sw/user/NGC_containers/pytorch_26.01-py3.sif && uv export --frozen --extra llm --extra dev --no-emit-project --no-hashes --no-header --no-annotate | grep -v -E '^(torch|torch-tensorrt|tensorrt[a-z0-9-]*|triton|nvidia-[a-z0-9-]+|cuda-[a-z-]+|numpy)==' > "$GAITLAB_STORE/ngc-reqs.txt" && apptainer exec "$SIF" "$GAITLAB_STORE/venv-ngc/bin/python" -m pip install --no-deps -r "$GAITLAB_STORE/ngc-reqs.txt" && apptainer exec "$SIF" "$GAITLAB_STORE/venv-ngc/bin/python" -m pip install --no-deps -e . && apptainer exec "$SIF" "$GAITLAB_STORE/venv-ngc/bin/python" -m pip check
   ```
4. **Jobs.** The same `rf_*` headers, with the `uv run …` line replaced by the line below. The result files record the container's versions through provenance, so they won't match `uv.lock`, and the README has to say which environment produced B2.

   ```bash
   apptainer exec --nv --bind "$HOME/gaitlab,$GAITLAB_STORE" "$SIF" "$GAITLAB_STORE/venv-ngc/bin/python" -m gaitlab.cli …
   ```

Write `rf_*_ngc.sbatch` variants only if this is needed, and review them before submitting.

---

## Later: after M2 (labels, splits) and the M4 benchmark code

8. **Real 1B LoRA.** The same job as step 5 without `--allow-synthetic` and `--max-steps`, on `data/sft`. Then `gaitlab predict --model 1b` and `gaitlab eval`. rsync `data/sft` up first; it holds features only, no video. `rsync-up.sh` has to be extended to allow it.
9. **Benchmarks.**
   - B2: `gaitlab bench --trt --model 1b` on the real test prompts with the real adapter, in the uv env if the gate's TensorRT lines passed, otherwise in the container.
   - B1: `bench/latency.py` `bench_pipeline`, once written. The latency clip `data/raw/BENCH/natural.mp4` is real video, so uploading it is your call; B2 alone needs no video.
   - Results name the exact GPU (`hardware.gpu` and `metrics.env.gpu_name`).

## Confirmed on DeltaAI (fill in at the end of the session)

| | Value |
|---|---|
| Login / compute OS, glibc | SLES 15 SP6, aarch64, glibc 2.38 on both (login: round 1; compute node gh116: `rf_gate` job 3534846, 2026-11-05) |
| NVIDIA driver, max CUDA (`nvidia-smi`) | driver 595.71.05, CUDA 13.2; GPU `NVIDIA GH200 120GB`, compute capability (9, 0) |
| torch, `torch.version.cuda`, cuDNN | torch 2.14.1+cu130, CUDA 13.0, cuDNN 9.24 (`92400`); Python 3.11.13 (system `/usr/bin/python3.11`), uv 0.13.7 |
| TensorRT, torch-tensorrt | 11.1.0.106, 2.14.0 from the uv env (`--extra trt` installs; no container needed). `import torch_tensorrt` needs a GPU (it queries the device at import), so it fails on login nodes by design |
| `GAITLAB_STORE`, venv, HF_HOME | `/work/nvme/bhhv/vwudaru/gaitlab` (set from round 1); venv `$GAITLAB_STORE/venv`, HF_HOME `$GAITLAB_STORE/hf` (Gemma 1B + 270M) |
| Python | **uv-managed CPython 3.11** (`UV_PYTHON_PREFERENCE=only-managed`). The system `/usr/bin/python3.11` has no `Python.h`, and Triton compiles `driver.c` against it on first use: `rf_synth` job 3536638 failed with `fatal error: Python.h: No such file or directory` |
| Compute-node internet | reachable (`rf_gate`: pypi.org:443). Jobs still run with `UV_OFFLINE=1 HF_HUB_OFFLINE=1` so nothing is fetched mid-run |
| Queue | `--nice=1000` floored priority to 1; `scontrol update jobid=<id> nice=0` (allowed for users) and a shorter `timelimit` on the pending job got it scheduled |
| 1B LoRA dry run (`rf_synth`) | COMPLETED, job 3539884 (2026-11-06): 20 steps on synthetic SFT, then `predict --synthetic`; run `1b-20261106T161038Z-b9c09dee`. Synthetic, never reported. The earlier job 3536638 failed on `Python.h` and left a partial `runs/1b-20261106T080301Z-b9c09dee` |
| `pytest -m gpu` on a GH200 | TBD. `test_cuda_time_on_gpu` passed (job 3536639); the TensorRT 270M test did not finish (jobs 3536639 time limit, 3539885 stopped by `timeout` inside `_probs` on the compiled module) |
| GPU-hours used (total, `budget.sh`) | TBD |

## Troubleshooting

| What you see | Fix |
|---|---|
| `env.sh: set GAITLAB_STORE ...` | round 1 isn't done: set it at the top of `scripts/cluster/env.sh` on the laptop, commit, re-run `rsync-up.sh` |
| `fatal error: Python.h: No such file or directory` from `triton/backends/nvidia/driver.c` | the venv was built on the system Python (no headers): `uv python install 3.11`, delete `$GAITLAB_STORE/venv`, `uv sync` again with `UV_PYTHON_PREFERENCE=only-managed` |
| `SYNC trt: FAILED` mentioning `manylinux_2_35` or "no wheel" | glibc < 2.35: no TensorRT in the uv env. LoRA steps still run; B2 needs "Fallback". |
| `torch.version.cuda` is `None` | a CPU-only torch: stop and paste the output (not expected from this lock) |
| 401 / 403 downloading Gemma | 401: redo `hf auth login`. 403: accept the Gemma license on huggingface.co, or allow gated repos on the token. |
| `OFFLINE` / cache error in a job | the model isn't in `$HF_HOME/hub`: redo `setup-login.sh models` |
| `uv: command not found` in a job | `env.sh` puts `~/.local/bin` on `PATH`; check that `setup-login.sh uv` ran |
| `NOT SUBMITTED` from `submit.sh` | you answered no at the 14.0 GH200-hour prompt (or a header check failed) |
| job stays `PD` | normal; `squeue --me --start` gives an estimate. `sprio -j <id>` shows the priority; if it is 1, a nice penalty is flooring it: `scontrol update jobid=<id> nice=0` |
| no log file | `submit.sh` creates `logs/`; submit from `~/gaitlab` with `submit.sh`, not plain `sbatch` |
