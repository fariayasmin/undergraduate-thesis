"""
Phase I0 - environment check and timing benchmark for Stage I-J.

Downloads TinyLlama-1.1B-Chat-v1.0 (if not already cached), then measures:
  - one forward pass at 1,500 input tokens (context-budget size, B)
  - one 60-token greedy generation
  - one training step at 512 tokens and one at 1,024 tokens, with a small
    residual-bottleneck adapter (matching Eqs. 56-57's shape) attached to
    every decoder layer so the timing reflects Phase I3's actual training
    shape. This benchmark script does NOT define the real stage_ij/adapter.py
    module (that is Phase I3's deliverable) - the class below is a minimal,
    throwaway stand-in used only to get a realistic timing number.

Device-agnostic: reads DEVICE from environment (default "cpu"); never calls
.cuda() directly. No bitsandbytes import anywhere.

Run with the stage_ij venv:
    stage_ij/.venv/Scripts/python.exe stage_ij/phase_i0_env_check.py
"""
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

import psutil
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = os.environ.get("STAGE_IJ_DEVICE", "cpu")
SEED = 42
OUT_DIR = Path(__file__).parent / "runs" / "i0_env_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def peak_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


class ThrowawayBottleneck(nn.Module):
    """
    Minimal stand-in for Eqs. (56)-(57), for TIMING ONLY. Not the Phase I3
    module. z = ReLU(h W_D); h' = h + z W_U. W_U starts at zero so an
    untrained adapter is the identity, matching the real design.
    """

    def __init__(self, hidden_size: int, bottleneck: int = 16):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, hidden_size, bias=False)
        nn.init.zeros_(self.up.weight)
        self.act = nn.ReLU()

    def forward(self, h):
        z = self.act(self.down(h))
        return h + self.up(z)


def main():
    torch.manual_seed(SEED)
    report = {}

    report["platform"] = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": DEVICE,
        "cpu_count_logical": os.cpu_count(),
    }
    report["hf_home"] = os.environ["HF_HOME"]

    print("=" * 78)
    print("Phase I0 - environment check")
    print("=" * 78)
    for k, v in report["platform"].items():
        print(f"  {k}: {v}")
    print(f"  HF_HOME: {report['hf_home']}")

    print("\nLoading tokenizer + model (downloads on first run)...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.to(DEVICE)
    load_s = time.time() - t0
    print(f"  loaded in {load_s:.1f}s")
    report["load_seconds"] = load_s
    report["n_params"] = sum(p.numel() for p in model.parameters())
    report["hidden_size"] = model.config.hidden_size
    report["n_layers"] = model.config.num_hidden_layers
    print(f"  params: {report['n_params']:,}  hidden={report['hidden_size']}  layers={report['n_layers']}")

    # ---- 1. forward pass at 1,500 tokens ----------------------------------
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, tok.vocab_size, (1, 1500), device=DEVICE)
    model.eval()
    with torch.no_grad():
        t0 = time.time()
        model(input_ids)
        fwd_s = time.time() - t0
    rss_after_fwd = peak_rss_mb()
    print(f"\n1. forward pass, 1500 tokens: {fwd_s:.3f}s  RSS={rss_after_fwd:.0f} MB")
    report["forward_1500_tokens_seconds"] = fwd_s
    report["rss_after_forward_mb"] = rss_after_fwd

    # ---- 2. 60-token greedy generation -------------------------------------
    prompt_ids = torch.randint(0, tok.vocab_size, (1, 200), device=DEVICE)
    with torch.no_grad():
        t0 = time.time()
        model.generate(prompt_ids, max_new_tokens=60, do_sample=False,
                       pad_token_id=tok.eos_token_id)
        gen_s = time.time() - t0
    print(f"2. greedy generation, 60 new tokens (200-token prompt): {gen_s:.3f}s")
    report["generate_60_tokens_seconds"] = gen_s

    # ---- 3. one training step at 512 and 1024 tokens, adapter attached ----
    print("\n3. training step timing (throwaway bottleneck adapter attached)")
    for p in model.parameters():
        p.requires_grad_(False)

    adapters = nn.ModuleList()
    hooks = []
    hidden = model.config.hidden_size
    layers = model.model.layers

    def make_hook(adapter):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                return (adapter(output[0]),) + output[1:]
            return adapter(output)
        return hook

    for layer in layers:
        ad = ThrowawayBottleneck(hidden, bottleneck=16)
        adapters.append(ad)
        hooks.append(layer.mlp.register_forward_hook(make_hook(ad)))

    trainable = [p for p in adapters.parameters()]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  trainable adapter params: {n_trainable:,} "
          f"({100 * n_trainable / report['n_params']:.3f}% of base model)")
    report["adapter_trainable_params_bench"] = n_trainable

    opt = torch.optim.AdamW(trainable, lr=1e-4)
    model.train()
    for seq_len in (512, 1024):
        torch.manual_seed(SEED)
        ids = torch.randint(0, tok.vocab_size, (1, seq_len), device=DEVICE)
        labels = ids.clone()
        t0 = time.time()
        opt.zero_grad()
        out = model(ids, labels=labels)
        out.loss.backward()
        opt.step()
        step_s = time.time() - t0
        rss = peak_rss_mb()
        print(f"  seq_len={seq_len}: {step_s:.3f}s/step  RSS={rss:.0f} MB  loss={out.loss.item():.3f}")
        report[f"train_step_{seq_len}_seconds"] = step_s
        report[f"rss_after_train_{seq_len}_mb"] = rss

    for h in hooks:
        h.remove()

    # ---- estimate total time for I2-I4 -------------------------------------
    # I2: zero-shot inference over ~ (1000 train + 5*150 val + 5*150 test)
    # items is not run in I2 (I2 only evaluates val/test-sized baseline sets;
    # train items are only generated in I1, not run through the model until
    # I3 training). Rough budget, stated explicitly as a rough estimate:
    n_baseline_items = 150 * 6 * 2  # val+test, ~6 categories, pooled+persistence+random+truncated (approx factor 2 for margin)
    est_i2_hours = (n_baseline_items * (fwd_s + gen_s)) / 3600
    steps_per_epoch = 1000 // 8  # batch 1, grad-accum 8 -> 8 examples/optim step
    est_i3_hours = (steps_per_epoch * 2 * report["train_step_1024_seconds"] * 8) / 3600
    report["estimate_i2_hours_rough"] = est_i2_hours
    report["estimate_i3_hours_rough"] = est_i3_hours
    print(f"\nRough estimate, I2 (zero-shot baselines over ~{n_baseline_items} items): "
          f"{est_i2_hours:.1f} h")
    print(f"Rough estimate, I3 (adapter training, 2 epochs, 1000 items, batch1/accum8): "
          f"{est_i3_hours:.1f} h")
    print("(I4's experiments are additional on top of this; sized once I1's data card is fixed.)")

    (OUT_DIR / "i0_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote {OUT_DIR / 'i0_report.json'}")


if __name__ == "__main__":
    main()
