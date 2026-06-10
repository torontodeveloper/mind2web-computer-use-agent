# Mind2Web Computer-Use Agent

An end-to-end computer-use (browser) agent trained on the [Mind2Web](https://osu-nlp-group.github.io/Mind2Web/) benchmark. The agent takes a natural-language task (e.g. *"Book a one-way flight from PIT to SFO"*), grounds it against a live web page, predicts the next action, and executes it in a real browser with Playwright.

## Architecture

```
Natural-language task
        │
        ▼
┌─────────────────────────┐
│ 1. Candidate Generation │  DeBERTa cross-encoder ranks DOM elements
│    (src/candidate_      │  → top-k candidate elements for the task
│     generation)         │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 2. Action Prediction    │  LLM (FLAN-T5 / GPT-5) selects the element
│    (src/action_         │  and operation: CLICK, TYPE, SELECT
│     prediction)         │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 3. Execution            │  Playwright drives a real browser:
│    (src/playwright)     │  locate element → act → observe new state
└─────────────────────────┘
            │ loop until task complete
            ▼
```

### Stage 1 — Candidate generation (DeBERTa)

A fine-tuned DeBERTa cross-encoder scores every interactable DOM element against the task description and action history, pruning thousands of nodes down to a small candidate set. See `src/candidate_generation/` (model, training, evaluation, Hydra configs in `conf/`).

### Stage 2 — Action prediction (LLM)

Given the top-k candidates, an LLM predicts the next action — which element to act on and the operation (CLICK / TYPE / SELECT) plus any value to type. Supports fine-tuned FLAN-T5 and API-based GPT-5 (`src/action_prediction/`, prompt template in `llm_prompt.json`).

### Stage 3 — Browser execution (Playwright)

`src/playwright/main.py` closes the loop: it loads the trained candidate model, snapshots the live DOM, runs the two-stage pipeline, and executes the predicted action in Chromium/Firefox/WebKit.

## Dataset

[Mind2Web](https://osu-nlp-group.github.io/Mind2Web/): 2,000+ tasks across 137 real websites, with full DOM snapshots, screenshots, and action traces. Generalization is evaluated on three splits: cross-task, cross-website, and cross-domain.

## Repository layout

```
src/
├── candidate_generation/   DeBERTa cross-encoder: train / evaluate / configs
├── action_prediction/      LLM action model: train / evaluate / LLM prompts
├── data_utils/             DOM processing, trace/snapshot preprocessing
└── playwright/             live browser execution of the trained pipeline
docs/                       presentation slides
```

## Training

Training runs on Google Colab (GPU) with data and checkpoints in Google Drive (`MyDrive/Mind2Web/`):

1. Preprocess data: `src/data_utils/` → `data_candidate_generation/` shards
2. Train candidate generator: `src/candidate_generation/train.py`
3. Train / configure action predictor: `src/action_prediction/train.py` (FLAN-T5) or use GPT-5 via `evaluate_llm.py`
4. Run the agent: `src/playwright/main.py`

## Roadmap

- [ ] Multimodal grounding: add page screenshots to action prediction (SeeAct-style) alongside DOM candidates
- [ ] Multi-step task loop with self-correction on failed actions
- [ ] Evaluation harness on the three Mind2Web generalization splits

## Acknowledgments

Built on the Mind2Web benchmark and baseline architecture from [OSU-NLP-Group/Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web) (MindAct). Developed as part of CMU coursework.
