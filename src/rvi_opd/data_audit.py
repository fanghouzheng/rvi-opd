from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping


_WHITESPACE = re.compile(r"\s+")


def normalize_prompt(prompt: str) -> str:
    """Conservative exact-dedup normalization; semantic/MinHash audit remains separate."""

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", prompt)).strip()


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(normalize_prompt(prompt).encode("utf-8")).hexdigest()


def build_prompt_manifest(
    rows: Iterable[Mapping[str, Any]], id_field: str, prompt_field: str
) -> Dict[str, Any]:
    records: List[Dict[str, str]] = []
    groups: Dict[str, List[str]] = defaultdict(list)
    seen_ids = set()
    for row_number, row in enumerate(rows, start=1):
        try:
            source_id = str(row[id_field])
            prompt = row[prompt_field]
        except KeyError as exc:
            raise ValueError(f"row {row_number} is missing field {exc.args[0]!r}") from exc
        if source_id in seen_ids:
            raise ValueError(f"duplicate source ID {source_id!r}")
        seen_ids.add(source_id)
        digest = prompt_sha256(prompt)
        records.append({"source_id": source_id, "normalized_prompt_sha256": digest})
        groups[digest].append(source_id)

    if not records:
        raise ValueError("no prompt rows")
    records.sort(key=lambda item: item["source_id"])
    duplicate_groups = [
        {"normalized_prompt_sha256": digest, "source_ids": sorted(source_ids)}
        for digest, source_ids in sorted(groups.items())
        if len(source_ids) > 1
    ]
    unsigned = {
        "physical_rows": len(records),
        "unique_normalized_prompts": len(groups),
        "duplicate_physical_rows": len(records) - len(groups),
        "records": records,
        "duplicate_groups": duplicate_groups,
        "normalization": "Unicode NFKC + whitespace collapse + strip",
    }
    manifest_sha = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**unsigned, "manifest_sha256": manifest_sha}
