# Stage I-J: Twin Adapter and LLM Explainability Layer

Adds an explainability layer over the frozen knowledge graph (Stages A-H):
a Twin Adapter (Eqs. 56-57) fine-tuned into TinyLlama-1.1B-Chat-v1.0, which
answers operator/policy questions about the twin's own stored KG facts. The
LLM explains the LP's decisions; it never makes dispatch decisions.

Stages A-H, their outputs, the population cache, and the LP objective are
untouched by this work. The only change to an earlier-stage file is
`08_knowledge_graph.py` gaining a `--kg-tag`/output-folder argument so each
tag builds into its own `outputs/kg{tag}/` (Phase I1).

## Environment

- Dedicated venv at `stage_ij/.venv`, Python 3.12.10 (CPU-only PyTorch).
- Pinned package versions in `stage_ij/requirements.txt` (see the setup
  commands at the top of that file).
- `HF_HOME` set explicitly to `C:\Users\C4S\.cache\huggingface`.
- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, loaded in float32, CPU only.
  No `.cuda()` calls anywhere; device comes from config/env
  (`STAGE_IJ_DEVICE`, default `cpu`). No `bitsandbytes` import anywhere in
  the installed dependency tree (verified: absent from
  `stage_ij/requirements.txt`).

## Reproducing Phase I0 (environment check + timing benchmark)

```
stage_ij/.venv/Scripts/python.exe stage_ij/phase_i0_env_check.py
```

Writes `stage_ij/runs/i0_env_check/i0_report.json`. Full results as run on
this machine:

**Hardware:** Windows 11 Pro, Intel Core i5-8365U @ 1.60GHz (4 cores / 8
logical), 15.82 GB RAM, 42 GB free disk (before model download; 38 GB after,
model cache ≈2.1 GB).

**Timings (CPU, float32, batch 1):**

| measurement | value |
|---|---|
| model load (first run, incl. download) | 904.9 s |
| forward pass, 1,500 tokens | 32.8 s, RSS 4,829 MB |
| greedy generation, 60 new tokens (200-token prompt) | 23.2 s |
| training step, seq_len=512, adapter attached | 23.2 s/step, RSS 4,742 MB |
| training step, seq_len=1,024, adapter attached | 49.9 s/step, RSS 4,683 MB |
| adapter trainable params (throwaway bench module, l=16, all 22 layers) | 1,441,792 (0.131% of the 1.1B base model) |

**Rough time estimates for later phases** (stated as rough; will be
refined once Phase I1's data card fixes the actual item counts):
- Phase I2 (zero-shot baselines over ≈1,800 val+test items, 4 non-oracle
  strategies): **≈28 hours** of wall-clock CPU time.
- Phase I3 (adapter training, 2 epochs, 1,000 training items, batch 1 /
  grad-accum 8, seq_len 1,024): **≈28 hours**.
- Phase I4's five experiments are additional on top of these two, and will
  be sized once I1/I2/I3's actual per-item costs are known from real runs
  rather than this single-step extrapolation.

These are substantial multi-day CPU budgets. Given the guiding principle to
prefer correctness over performance and not optimise prematurely, Phase I1's
dataset sizes (1,000 train / up to 150 per category for val and test) are
built as specified; if I2/I3's real measured costs confirm this rough
estimate, that is a decision point to raise before committing the wall-clock
time, not something to silently shrink or skip.

## Design decisions

**I0.1 - Python 3.12, not the system default 3.14.** The machine's default
`python` resolves to 3.14.6, which is too new for reliable CPU-wheel
availability across the full torch/transformers/peft/accelerate/datasets
stack at the time of this run. Python 3.12.10 (already installed on this
machine via the `py` launcher) is the most conservative choice with broad
ML-library wheel support, so `stage_ij/.venv` was created against it
explicitly rather than the shell's default `python`.

**I0.2 - torch installed from the CPU wheel index, not plain PyPI.**
`pip install torch` from plain PyPI resolves to whatever build the resolver
picks first, which is not guaranteed to be the CPU-only build and can pull
CUDA dependencies that are useless (and large) on this machine. Installing
explicitly via `--index-url https://download.pytorch.org/whl/cpu` guarantees
the `+cpu` build (installed: `torch==2.14.0+cpu`). `requirements.txt`
documents this as a required two-step install (torch first, from the CPU
index; everything else via `--no-deps` against the frozen list) so a fresh
setup does not accidentally re-resolve a CUDA torch build.

**I0.3 - `HF_HOME` set explicitly rather than left at the OS default.**
The task requires the model cache to live under the user's own directory in
a location that is documented and reproducible, rather than whatever
`huggingface_hub`'s own default resolves to. Set to
`C:\Users\C4S\.cache\huggingface` (a conventional location, but pinned via
an explicit environment variable rather than relying on the default).

**I0.4 - the I0 benchmark's adapter is a throwaway stand-in, not
`stage_ij/adapter.py`.** Phase I0 needs a realistic training-step timing
number, which requires SOME adapter attached to get realistic
memory/compute shape, but Phase I3 is the assigned owner of the real
Twin-Adapter module (Eqs. 56-57). `phase_i0_env_check.py` defines a minimal,
local `ThrowawayBottleneck` class purely for this timing measurement, so
Phase I3 is not pre-empted and there is nothing to "rewrite" later - I3 will
write the real module from scratch against the formulation, independent of
this benchmark's throwaway version.

**I0.5 - symlink warning from `huggingface_hub`, left as a warning, not
fixed.** Windows without Developer Mode does not support the symlink-based
HF cache layout; `huggingface_hub` degrades gracefully to plain file copies
(confirmed: the model still downloaded and loaded correctly). Enabling
Developer Mode or running as Administrator would silence the warning but is
an OS-level change outside this stage's scope, and the degraded mode has no
observed functional effect other than slightly higher disk use from lack of
deduplication - not worth the OS change for a research repo.
