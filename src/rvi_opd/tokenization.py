from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class LexiconArtifact:
    name: str
    tokenizer_revision: str
    tokenizer_sha256: str
    vocabulary_sha256: str
    phrase_variants: Tuple[str, ...]
    token_sequences: Dict[str, Tuple[int, ...]]
    first_subword_ids: Tuple[int, ...]
    artifact_sha256: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def build_lexicon_artifact(
    name: str,
    phrases: Iterable[str],
    encode: Callable[[str], Sequence[int]],
    tokenizer_revision: str,
    tokenizer_sha256: str,
    vocabulary_sha256: str,
    include_lowercase: bool,
) -> LexiconArtifact:
    """Resolve bare/leading-space (and optionally lowercase) tokenizer variants."""

    variants: List[str] = []
    for phrase in phrases:
        bases = [phrase]
        if include_lowercase and phrase.lower() != phrase:
            bases.append(phrase.lower())
        for base in bases:
            variants.extend([base, " " + base])
    variants = list(dict.fromkeys(variants))
    if not variants:
        raise ValueError("phrases must not be empty")

    sequences: Dict[str, Tuple[int, ...]] = {}
    first_ids = set()
    for variant in variants:
        token_ids = tuple(int(token_id) for token_id in encode(variant))
        if not token_ids:
            raise ValueError(f"tokenizer returned an empty sequence for {variant!r}")
        sequences[variant] = token_ids
        first_ids.add(token_ids[0])

    unsigned = {
        "name": name,
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_sha256": tokenizer_sha256,
        "vocabulary_sha256": vocabulary_sha256,
        "phrase_variants": variants,
        "token_sequences": {key: list(value) for key, value in sequences.items()},
        "first_subword_ids": sorted(first_ids),
    }
    digest = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return LexiconArtifact(
        name=name,
        tokenizer_revision=tokenizer_revision,
        tokenizer_sha256=tokenizer_sha256,
        vocabulary_sha256=vocabulary_sha256,
        phrase_variants=tuple(variants),
        token_sequences=sequences,
        first_subword_ids=tuple(sorted(first_ids)),
        artifact_sha256=digest,
    )
