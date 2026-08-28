# Synthetic examples

These files contain invented scalar signals only—no benchmark prompt, response or model output.

```bash
rvi-opd route-jsonl \
  --calibration examples/calibration_states.jsonl \
  --input examples/states_to_route.jsonl \
  --output runs/example/routes.jsonl

# Replay routing without recalibrating D1:
rvi-opd route-jsonl \
  --threshold-artifact runs/example/routes.thresholds.json \
  --input examples/states_to_route.jsonl \
  --output runs/example/routes-replayed.jsonl

rvi-opd audit-budget \
  --ledger examples/budget_ledger.jsonl \
  --left-arm repair \
  --right-arm intervene \
  --match-on examples,teacher_scored_tokens,student_supervised_tokens,optimizer_steps

rvi-opd audit-prompts \
  --input examples/prompts.jsonl \
  --output runs/example/prompt_manifest.json
```

Use `--repetition-threshold 0.8` and/or `--paced-zero-rescue` only for D4/D5 boundary experiments; both are disabled in the confirmatory core.

The prompt manifest contains IDs and hashes, not prompt text. It catches physical exact duplicates after conservative Unicode/whitespace normalization; MinHash, equation signatures and semantic review remain mandatory for the full data audit.
