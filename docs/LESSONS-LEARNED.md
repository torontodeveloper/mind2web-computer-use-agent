# Debugging Log: NaN Logits in DeBERTa Candidate-Generation Training

**Date:** 2026-06-10
**Symptom:** Training ran but produced NaN logits on **every step**, so every backward
pass was skipped — the model trained for thousands of steps while learning nothing.

---

## Timeline of failures (each one a separate bug)

### Bug 1 — `FileNotFoundError: No shard files found in .../Mind2Web/...`
- **Cause:** project folder was renamed `Mind2Web` → `mind2web-computer-use-agent`,
  but `dataloader.py`, `train.py`, and Hydra configs had the old path **hardcoded**.
  The `data.data_path` Hydra override was passed but `get_data_split()` ignored it
  and used its own hardcoded `split_dir_map`.
- **Fix:** updated all hardcoded paths in `src/` (commit `032b921`).
- **Lesson:** paths belong in config (one place), not scattered through code.
  `dataloader.py` should respect `data.data_path` instead of hardcoding.

### Bug 2 — `UnboundLocalError: cannot access local variable 'logits'`
- **Cause:** leftover debug prints from an earlier NaN-hunting session were pasted
  **above** the forward pass in `model.py fit()`, referencing `logits` / `loss_value`
  before they were computed. Crashed on the first training step.
- **Fix:** removed premature prints; kept NaN guards; limited verbose prints to the
  first 3 steps (commit `cefe18d`).
- **Lesson:** debug prints that reference variables only valid *after* a statement
  must live after it; clean up debug scaffolding before the next run.

### Bug 3 (the big one) — NaN logits on 100% of training steps
- **Symptom:** `⚠️ NaN in logits, skipping backward` on every step from step 0.
  Inputs were verified NaN-free; weights were fresh `microsoft/deberta-v3-base`;
  AMP was off; loss was BCEWithLogitsLoss.
- **What we ruled out:**
  - Exploding gradients / bad LR — impossible, no optimizer step ever ran (NaN from step 0).
  - Bad input data — input NaN checks never fired.
  - SDPA attention bug — DeBERTa-v2/v3 **does not support SDPA at all**
    (transformers raises `ValueError`), so training was already running eager attention.
  - Plain transformers forward — `AutoModelForSequenceClassification` +
    deberta-v3-base on transformers **5.10.1** gives a finite logit. The base
    library is fine.
- **Root cause (confirmed via dtype check):** the sentence-transformers
  CrossEncoder path loads the model in **half precision by default** on newer
  library versions. The diagnostic logit `0.03533935546875` is exactly
  representable in fp16 — a fingerprint of half-precision compute.
  **DeBERTa-v3's disentangled attention overflows fp16** (large intermediate
  values), producing NaN on realistic batch sizes/sequence lengths while tiny
  test inputs may survive.
- **Contributing factor:** version drift. Our custom `CrossEncoder.fit()` in
  `model.py` is copied from sentence-transformers **2.x** (see the
  `_target_device` deprecation warning in the training log), but Colab installed
  the latest sentence-transformers. APIs and defaults (including dtype) changed
  under us.
- **Fix:** pin known-good versions in the notebook pip cell, restart runtime, retrain:
  ```
  pip install "transformers==4.44.2" "sentence-transformers==2.7.0" \
              datasets lxml hydra-core omegaconf
  ```
  (Alternative: keep latest libs but force `torch_dtype=torch.float32` when the
  CrossEncoder constructs the model — and eventually rebase our subclass onto the
  current sentence-transformers API.)

---

## Process lessons

1. **Fail fast beats fail silent.** The NaN-skip guard let training "run" for
   2,600+ steps while learning nothing. TODO: abort after N consecutive NaN steps
   (e.g. 50) instead of skipping forever.
2. **Pin your dependencies.** Colab gives you a fresh latest-everything runtime
   every session. `requirements.txt` exists but the notebook's bare
   `pip install datasets sentence-transformers ...` ignores pins. A working run
   today is not a working run next week.
3. **fp16 is not free.** Half precision is a numerically meaningful change, not
   just a speed knob — some architectures (DeBERTa's disentangled attention,
   some attention-softmax patterns) overflow fp16. bf16 has fp32's range and
   avoids this class of bug; or stay fp32 for a 86M-param model where memory
   isn't the constraint.
4. **`logging` over `print` for training.** Hydra tees `logging` output to
   `train.log` on Drive, which survives runtime death and output truncation.
   `print()` only reaches the (truncatable) cell output.
5. **One source of truth for code.** The stale-Drive-copy episode (Colab ran
   June-4 code while local was fixed) cost a failed run. Better workflow:
   code lives on GitHub, Colab does `git clone`/`git pull` into `/content`,
   Drive holds only data + checkpoints.

## Interview-ready story (60 seconds)

> "My fine-tuning run was silently broken: 100% of steps produced NaN logits and
> the NaN-guard skipped every backward pass, so it 'trained' while learning
> nothing. I ruled out data (input NaN checks), optimization (NaN from step 0,
> before any update), and the base library (plain transformers forward was
> finite). The fingerprint was the logit value being exactly fp16-representable —
> a library upgrade had changed the default load dtype to half precision, and
> DeBERTa-v3's disentangled attention overflows fp16. Pinning library versions
> and forcing fp32 fixed it. My takeaways: pin dependencies in ephemeral
> environments, fail fast on NaN instead of skipping, and know your
> architecture's numerical limits — DeBERTa can't do fp16, and it can't do
> FlashAttention/SDPA either."

## Open TODOs

- [x] Confirmed in Colab 2026-06-10: `model dtype: torch.float16` — root cause verified
- [ ] Pin versions in notebook pip cell + retrain
- [ ] Add consecutive-NaN abort to `model.py fit()`
- [ ] Make `dataloader.py` respect `data.data_path` (remove hardcoded split_dir_map)
- [ ] Move per-step `use_amp` print outside the loop
- [ ] Notebook: skip dataset download if shards already exist
- [ ] Switch Colab workflow to `git pull` instead of Drive-synced code
