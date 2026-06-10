import json
import asyncio
import torch
import sys
import os
from pathlib import Path
from playwright.async_api import async_playwright


# sys.path.append("/content/drive/MyDrive/Mind2Web/src/candidate_generation")
HERE = Path(__file__).resolve()  # src/playwright/main.py
ROOT = HERE.parent.parent
sys.path.append(str(ROOT))
from candidate_generation.model import CrossEncoder

device = torch.device("cpu")

# Checkpoint location: override with MIND2WEB_CKPT, else use the Drive-synced
# checkpoints written by the Colab training notebook.
DEFAULT_CKPT = (
    Path.home()
    / "My Drive (kkakolla@andrew.cmu.edu)"
    / "mind2web-computer-use-agent"
    / "checkpoints"
    / "candidate_generation"
)
CKPT_PATH = os.environ.get("MIND2WEB_CKPT", str(DEFAULT_CKPT))

candidate_model = CrossEncoder(
    CKPT_PATH,
    device=device,
    num_labels=1,
    max_length=512,
)
import torch

nan_param_names = []
for name, param in candidate_model.model.named_parameters():
    if torch.isnan(param).any():
        nan_param_names.append(name)

print("Num params with NaNs:", len(nan_param_names))
print("First few:", nan_param_names[:10])

# ── Format functions must match dataloader.py exactly ────────


def format_candidate_from_dom(el: dict) -> str:
    """Reproduce format_candidate_simple() from dataloader.py using live DOM data."""
    parts = []

    if el.get("tag"):
        parts.append(f"tag: {el['tag']}")
    if el.get("backend_node_id"):
        parts.append(f"candidate_id: {el['backend_node_id']}")

    # Map DOM attributes to the keys format_candidate_simple() looks for
    attr_map = {
        "role": el.get("role", ""),
        "type": el.get("type", ""),
        "name": el.get("name", ""),
        "title": el.get("title", ""),
        "aria_label": el.get("aria_label", ""),  # note: underscore not hyphen
        "placeholder": el.get("placeholder", ""),
        "value": el.get("value", ""),
        "is_clickable": el.get("is_clickable", ""),
        "bounding_box_rect": el.get("bounding_box_rect", ""),
    }

    for key, value in attr_map.items():
        if value not in (None, ""):
            parts.append(f"{key}: {value}")

    return " | ".join(parts) if parts else str(el)


def build_query(task: str, previous_actions: list[str]) -> str:
    """Reproduce the query format from CandidateRankDataset.__getitem__()."""
    previous = "; ".join(previous_actions[-3:]) if previous_actions else ""
    return f"task is: {task}\n" f"Previous actions: {previous}"


# ── Extract elements from live DOM, matching Mind2Web fields ─


async def extract_dom_elements(page) -> list[dict]:
    """Extract elements with the same attributes Mind2Web stored."""
    return await page.evaluate(
        """
    () => Array.from(document.querySelectorAll(
       'a, button, input, select, textarea, [role=button], [role=link], [role=checkbox], [role=menuitem], [role=option], [role=tab], [role=combobox]' 
    ))
    .filter(el => {
    const r = el.getBoundingClientRect();
    const text = (el.innerText || el.textContent || '').trim().toLowerCase();

    if (r.width <= 0 || r.height <= 0) return false;
    if (r.bottom < 0 || r.right < 0) return false;

    if (text === 'skip to main content') return false;
    if (text.startsWith('skip to')) return false;
    if (text.includes('accessibility')) return false;

    if (!text &&
    !el.getAttribute('aria-label') &&
    !el.getAttribute('placeholder') &&
    el.tagName !== 'INPUT') return false;

    return true;
})
    .map((el, idx) => {
        const r = el.getBoundingClientRect();
        return {
            backend_node_id: String(idx),   // approximation
            tag:         el.tagName.toLowerCase(),
            role:        el.getAttribute('role') || '',
            type:        el.getAttribute('type') || '',
            name:        el.getAttribute('name') || '',
            title:       el.getAttribute('title') || '',
            aria_label:  el.getAttribute('aria-label') || '',
            placeholder: el.getAttribute('placeholder') || '',
            value:       el.value || '',
            is_clickable: (el.tagName === 'A' || el.tagName === 'BUTTON' ||
                           el.onclick !== null || el.getAttribute('role') === 'button')
                          ? 'true' : '',
            bounding_box_rect: `${r.x},${r.y},${r.width},${r.height}`,
            // keep for selector-building later
            text:        (el.innerText || el.textContent || '').trim().slice(0, 100),
            x: r.x, y: r.y, width: r.width, height: r.height,
        };
    })
    """
    )


# ── CrossEncoder inference ────────────────────────────────────


def score_candidates(
    elements: list[dict], task: str, previous_actions: list[str], top_k=50
) -> list[dict]:
    query = build_query(task, previous_actions)

    # Order matches training: texts=[candidate, query]
    pairs = [(format_candidate_from_dom(el), query) for el in elements]
    test_pairs = [
        (
            "tag: input | candidate_id: 1 | aria_label: Destination | placeholder: To? | text: To?",
            "task is: Search for a flight from Toronto to New York. Type NYC into the destination field.\nPrevious actions: ",
        ),
        (
            "tag: button | candidate_id: 2 | aria_label: Search | text: Search | is_clickable: true",
            "task is: Search for a flight from Toronto to New York. Type NYC into the destination field.\nPrevious actions: ",
        ),
    ]

    scores = candidate_model.predict(test_pairs)  # numpy array shape (N,)
    print("Number of pairs:", len(pairs))
    print("First pair candidate text:", pairs[0][0][:300] if pairs else "NONE")
    print("First pair query:", pairs[0][1] if pairs else "NONE")
    print("Raw scores type:", type(scores))
    print("First 10 raw scores:", scores[:10] if len(scores) >= 10 else scores)
    scored = [{**el, "_score": float(s)} for el, s in zip(elements, scores)]
    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored[:top_k]


# ── Flan-T5 action prediction ─────────────────────────────────

from transformers import T5ForConditionalGeneration, T5Tokenizer

t5_tokenizer = T5Tokenizer.from_pretrained(
    "osunlp/MindAct_ActionPrediction_flan-t5-base"
)
t5_model = T5ForConditionalGeneration.from_pretrained(
    "osunlp/MindAct_ActionPrediction_flan-t5-base"
)
t5_model.eval()


def predict_action_t5(
    candidates: list[dict], task: str, previous_actions: list[str]
) -> dict:
    cand_str = "\n".join(
        f"[{i}] {format_candidate_from_dom(c)}" for i, c in enumerate(candidates)
    )
    # MindAct prompt format
    prompt = (
        f"task is: {task}\n"
        f"Previous actions: {'; '.join(previous_actions[-3:])}\n"
        f"Candidates:\n{cand_str}\n"
        f"Action:"
    )
    enc = t5_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        out = t5_model.generate(**enc, max_new_tokens=64)
    raw = t5_tokenizer.decode(out[0], skip_special_tokens=True).strip()
    print(f"  T5 raw: '{raw}'")

    parts = raw.strip().split()
    try:
        idx = int(parts[0])
        op = parts[1].upper() if len(parts) > 1 else "CLICK"
        value = " ".join(parts[2:]) if len(parts) > 2 else ""
    except (ValueError, IndexError):
        return None

    el = candidates[idx] if 0 <= idx < len(candidates) else candidates[0]
    return {"element": el, "operation": op, "value": value}


# ── Selector builder ─────────────────────────────────────────


def build_selector(el: dict) -> str:
    if el.get("aria_label"):
        return f"[aria-label='{el['aria_label']}']"
    if el.get("placeholder"):
        return f"[placeholder='{el['placeholder']}']"
    if el.get("name"):
        return f"[name='{el['name']}']"
    if el.get("text") and el["tag"] in ("button", "a"):
        return f"{el['tag']}:has-text('{el['text'][:40]}')"
    return el.get("tag", "div")


# ── Agent loop ───────────────────────────────────────────────


# async def run_agent(task: str, start_url: str, max_steps: int = 15):
#     previous_actions: list[str] = []

#     async with async_playwright() as p:
#         browser = await p.chromium.launch(headless=False, slow_mo=100)
#         page = await browser.new_page()
#         await page.goto(start_url, wait_until="domcontentloaded")
#         await page.wait_for_timeout(3000)

#         for step in range(max_steps):
#             print(f"\n── Step {step + 1} ──────────────────────────────")

#             # Stage 1: score all DOM elements with DeBERTa
#             elements = await extract_dom_elements(page)
#             print(f"  DOM elements: {len(elements)}")

#             candidates = score_candidates(elements, task, previous_actions, top_k=50)
#             for index, candidate in enumerate(candidates[:10]):
#                 print(
#                     f"[{index}] tag={candidate['tag']} text='{candidate.get('text','')[:50]}' "
#                     f"aria='{candidate.get('aria_label','')}' placeholder='{candidate.get('placeholder','')}' "
#                     f"score={candidate['_score']:.4f}"
#                 )
#             print(
#                 f"  Top candidate: [{candidates[0]['tag']}] aria='{candidates[0].get('aria_label','')}' text='{candidates[0].get('text','')[:40]}'"
#             )

#             # Stage 2: predict action with Flan-T5
#             action = predict_action_t5(candidates, task, previous_actions)
#             if action is None:
#                 print(f"Sorry T5 DidNOT predict Any Valid Action")
#                 break
#             el = action["element"]
#             op = action["operation"]
#             value = action["value"]
#             selector = build_selector(el)
#             print(f"  → {op} | selector='{selector}' | value='{value}'")

#             if op in ("SUBMIT", "DONE", "STOP"):
#                 print(f" HURRAY, YAY, Task complete!****")
#                 break

#             # Execute action
#             try:
#                 if op == "CLICK":
#                     await page.locator(selector).first.click(timeout=5000)
#                 elif op == "TYPE":
#                     await page.locator(selector).first.fill(value, timeout=5000)
#                 elif op == "SELECT":
#                     await page.locator(selector).first.select_option(
#                         value, timeout=5000
#                     )
#             except Exception as e:
#                 print(f"  ⚠ Selector failed: {e} — falling back to coordinates")
#                 cx = el.get("x", 0) + el.get("width", 10) / 2
#                 cy = el.get("y", 0) + el.get("height", 10) / 2
#                 await page.mouse.click(cx, cy)

#             # Track action history (format matches action_reprs in training data)
#             previous_actions.append(f"{op} {el.get('tag','')} {el.get('text','')[:30]}")
#             await page.wait_for_timeout(1500)

#         await page.screenshot(path="final_state.png")
#         await browser.close()


# asyncio.run(
#     run_agent(
#         task="Search for a flight from Toronto to New York. Type NYC into the destination field.",
#         start_url="https://www.ca.kayak.com/?ispredir=true",
#         max_steps=10,
#     )
# )

test_pairs = [
    (
        "tag: input | candidate_id: 1 | aria_label: Destination | placeholder: To? | text: To?",
        "task is: Search for a flight from Toronto to New York. Type NYC into the destination field.\nPrevious actions: "
    ),
    (
        "tag: button | candidate_id: 2 | aria_label: Search | text: Search | is_clickable: true",
        "task is: Search for a flight from Toronto to New York. Type NYC into the destination field.\nPrevious actions: "
    ),
]

print("ABOUT TO RUN SYNTHETIC TEST")
test_scores = candidate_model.predict(test_pairs)
print("SYNTHETIC TEST SCORES:", test_scores)
sys.exit()