from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, Tuple


@dataclass(frozen=True)
class BackendCapabilities:
    full_vocabulary_scores: bool
    arbitrary_token_gather: bool
    teacher_bridge_generation: bool
    deterministic_seed_pairing: bool
    kv_snapshot_restore: bool


@dataclass(frozen=True)
class ScoredState:
    state_id: str
    context_token_ids: Tuple[int, ...]
    teacher_probabilities: Mapping[int, float]
    student_probabilities: Mapping[int, float]


@dataclass(frozen=True)
class BridgeResult:
    state_id: str
    bridge_token_ids: Tuple[int, ...]
    paragraph_count: int
    stopped_by: str


@dataclass(frozen=True)
class Continuation:
    state_id: str
    seed: int
    token_ids: Tuple[int, ...]
    verifier_pass: bool


class ModelBackend(Protocol):
    tokenizer_revision: str
    tokenizer_sha256: str
    vocabulary_sha256: str
    capabilities: BackendCapabilities

    def score_states(self, state_ids: Sequence[str]) -> Sequence[ScoredState]: ...

    def generate_bridge(
        self, state_id: str, paragraphs: int, max_tokens: int, seed: int
    ) -> BridgeResult: ...

    def continue_student(
        self, state_id: str, context_token_ids: Sequence[int], max_tokens: int, seed: int
    ) -> Continuation: ...

    def restore_context(self, state_id: str) -> Sequence[int]: ...
