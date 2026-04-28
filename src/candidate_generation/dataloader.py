import json
import pathlib
import random
import sys
import glob

from sentence_transformers import InputExample
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.append(pathlib.Path(__file__).parent.parent.absolute().as_posix())


def format_candidate_simple(candidate):
    """
    Build a text representation directly from the stored candidate object.

    Expected candidate format:
    {
        "tag": "a",
        "attributes": "{\"backend_node_id\": \"39558\", ...}",
        "backend_node_id": "39558",
        ...
    }
    """
    tag = candidate.get("tag", "")
    backend_node_id = candidate.get("backend_node_id", "")

    attrs = candidate.get("attributes", {})
    if isinstance(attrs, str):
        try:
            attrs = json.loads(attrs)
        except json.JSONDecodeError:
            attrs = {"raw_attributes": attrs}
    elif not isinstance(attrs, dict):
        attrs = {}

    useful_keys = [
        "backend_node_id",
        "role",
        "type",
        "name",
        "class",
        "title",
        "alt",
        "aria_label",
        "placeholder",
        "value",
        "input_value",
        "input_checked",
        "is_clickable",
        "bounding_box_rect",
    ]

    parts = []
    if tag:
        parts.append(f"tag: {tag}")
    if backend_node_id:
        parts.append(f"candidate_id: {backend_node_id}")

    for key in useful_keys:
        value = attrs.get(key)
        if value not in (None, ""):
            parts.append(f"{key}: {value}")

    # fallback in case the candidate is very sparse
    if not parts:
        parts.append(str(candidate))

    return " | ".join(parts)


class CandidateRankDataset(Dataset):
    def __init__(self, data=None, neg_ratio=5):
        self.data = data or []
        self.neg_ratio = neg_ratio

    def __len__(self):
        return len(self.data) * (1 + self.neg_ratio)

    def __getitem__(self, idx):
        sample = self.data[idx // (1 + self.neg_ratio)]

        # Guard against malformed rows
        pos_candidates = sample.get("pos_candidates", [])
        neg_candidates = sample.get("neg_candidates", [])

        if len(pos_candidates) == 0:
            # move to the next sample cyclically
            idx = (idx + (1 + self.neg_ratio)) % len(self)
            return self.__getitem__(idx)

        if idx % (1 + self.neg_ratio) == 0 or len(neg_candidates) == 0:
            candidate = random.choice(pos_candidates)
            label = 1
        else:
            candidate = random.choice(neg_candidates)
            label = 0

        confirmed_task = sample.get("confirmed_task", "")
        action_reprs = sample.get("action_reprs", [])
        previous_actions = "; ".join(action_reprs[-3:]) if action_reprs else ""

        query = (
            f"task is: {confirmed_task}\n"
            f"Previous actions: {previous_actions}"
        )

        return InputExample(texts=[candidate[1], query], label=label)


def get_data_split(split_name="train", max_examples=9999):
    split_dir_map = {
        "train": "/content/drive/MyDrive/Mind2Web/data_candidate_generation/train",
        "test_website": "/content/drive/MyDrive/Mind2Web/data_candidate_generation/test_website",
        "test_task": "/content/drive/MyDrive/Mind2Web/data_candidate_generation/test_task",
        "test_domain": "/content/drive/MyDrive/Mind2Web/data_candidate_generation/test_domain",
    }
    shard_dir = split_dir_map.get(split_name)
    shard_files = sorted(glob.glob(f"{shard_dir}/shard_*.json"))

    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {shard_dir}")

    print(f"Loading {split_name} from {len(shard_files)} shards in {shard_dir}")

    raw_examples = []
    for shard_file in shard_files:
        with open(shard_file, "r") as f:
            raw_examples.extend(json.load(f))
        if len(raw_examples) >= max_examples:
            break

    raw_examples = raw_examples[:max_examples]
    print(f"Loaded {len(raw_examples)} examples from {split_name}")

    formatted = []
    skipped = 0

    MAX_NEGS = 20  # keep only 20 random negatives per sample

    for sample in tqdm(raw_examples[:max_examples], desc=f"Formatting {split_name}"):
        try:
            pos_cands = sample.get("pos_candidates", [])
            neg_cands = sample.get("neg_candidates", [])

            if pos_cands and isinstance(pos_cands[0], str):
                pos_cands = [json.loads(c) for c in pos_cands]
            if neg_cands and isinstance(neg_cands[0], str):
                neg_cands = [json.loads(c) for c in neg_cands]

            if len(pos_cands) == 0 and len(neg_cands) == 0:
                skipped += 1
                continue

            # Subsample negatives
            if len(neg_cands) > MAX_NEGS:
                neg_cands = random.sample(neg_cands, MAX_NEGS)

            sample["pos_candidates"] = [
                (c.get("backend_node_id", ""), format_candidate_simple(c))
                for c in pos_cands
            ]
            sample["neg_candidates"] = [
                (c.get("backend_node_id", ""), format_candidate_simple(c))
                for c in neg_cands
            ]
            formatted.append(sample)

        except Exception:
            skipped += 1
            continue

    print(f"Formatted {len(formatted)} examples, skipped {skipped}")

    return formatted