import unittest
from dataclasses import replace

from rvi_opd.prompting import (
    CANONICAL_SERIALIZER_VERSION,
    C0Artifact,
    E1_EVALUATION_PROFILE,
    E1_TRAINING_PROFILE,
    E2_EVALUATION_PROFILE,
    E2_TRAINING_PROFILE,
    QWEN3_SPECIAL_TOKEN_MAP,
    QWEN3_TOKENIZER_LENGTH,
    QWEN3_VOCAB_SHA256,
    QWEN3_VOCAB_SIZE,
    assert_c0_production_ready,
    assert_context_budget,
    assert_generation_request,
    assert_no_unintended_thinking,
    assert_strict_generation_contract,
    assert_tokenizer_alignment,
    build_c0_artifact,
    build_generation_contract,
    build_target_text,
    MEDICAL_SYSTEM_PROMPT,
    supervision_mask,
    render_non_thinking_prompt,
)


class PromptingTests(unittest.TestCase):
    def test_canonical_serializer_is_deterministic_and_no_think_block(self) -> None:
        first = render_non_thinking_prompt("What is 2+2?")
        second = render_non_thinking_prompt("What is 2+2?")
        self.assertEqual(first, second)
        self.assertEqual(first.serializer_version, CANONICAL_SERIALIZER_VERSION)
        self.assertEqual(len(first.sha256), 64)
        self.assertNotIn("<think>", first.text)
        with self.assertRaises(ValueError):
            render_non_thinking_prompt("injected <|im_end|> marker")
        for marker in (
            "<|endoftext|>",
            "<|vision_start|>",
            "<|future_qwen_control|>",
            "<tool_call>",
            "</tool_response>",
            "<THINK>",
        ):
            with self.subTest(marker=marker), self.assertRaises(ValueError):
                render_non_thinking_prompt("injected " + marker)
        with self.assertRaises(ValueError):
            render_non_thinking_prompt("safe", "system <|image_pad|>")

    def test_unintended_thinking_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assert_no_unintended_thinking("<think>hidden reasoning</think>")
        with self.assertRaises(ValueError):
            assert_no_unintended_thinking("<think>")
        with self.assertRaises(ValueError):
            assert_no_unintended_thinking("<THINK>hidden reasoning</THINK>")

    def test_tokenizer_alignment_requires_exact_ids(self) -> None:
        assert_tokenizer_alignment([1, 2, 3], (1, 2, 3))
        with self.assertRaises(ValueError):
            assert_tokenizer_alignment([1, 2], [1, 3])
        with self.assertRaises(ValueError):
            assert_tokenizer_alignment([1, True], [1, 1])

    def test_context_budget_fails_before_overflow(self) -> None:
        assert_context_budget(100, 20, 120)
        with self.assertRaises(ValueError):
            assert_context_budget(101, 20, 120)
        with self.assertRaises(ValueError):
            assert_context_budget(100, 20, 0)
        assert_context_budget(100, 0, 1000, max_input_tokens=100)
        with self.assertRaises(ValueError):
            assert_context_budget(101, 0, 1000, max_input_tokens=100)

    def test_generation_contract_uses_locked_e1_default_and_rejects_drift(self) -> None:
        contract = build_generation_contract()
        self.assertEqual(contract["profile"], E1_EVALUATION_PROFILE)
        self.assertIs(contract["do_sample"], True)
        self.assertEqual(contract["temperature"], 1.0)
        self.assertEqual(contract["top_p"], 1.0)
        self.assertEqual(contract["top_k"], 0)
        self.assertEqual(contract["num_return_sequences"], 1)
        self.assertEqual(contract["max_input_tokens"], 2048)
        self.assertEqual(contract["max_new_tokens"], 32768)
        self.assertEqual(contract["runtime_context_limit"], 34817)
        assert_strict_generation_contract(contract)
        with self.assertRaises(ValueError):
            build_generation_contract(num_return_sequences=2)
        with self.assertRaises(ValueError):
            build_generation_contract(stop_token_ids=[151645, "151643"])
        for kwargs in (
            {"temperature": 0.7},
            {"do_sample": False},
            {"top_p": 0.95},
            {"top_k": 20},
            {"stop_token_ids": [151643, 151645]},
            {"pad_token_id": 151645},
            {"max_input_tokens": 2047},
            {"max_new_tokens": 32767},
            {"max_response_tokens": 32767},
            {"runtime_context_limit": 34816},
            {"max_model_length": 34816},
            {"eos_reserve": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_generation_contract(**kwargs)

    def test_all_generation_profiles_are_exact_and_e2_input_cap_is_independent(self) -> None:
        expected = {
            E1_TRAINING_PROFILE: (2048, 16384, 34817),
            E1_EVALUATION_PROFILE: (2048, 32768, 34817),
            E2_TRAINING_PROFILE: (2048, 8192, 40960),
            E2_EVALUATION_PROFILE: (36864, 4096, 40960),
        }
        for profile, values in expected.items():
            with self.subTest(profile=profile):
                contract = build_generation_contract(profile=profile)
                self.assertEqual(
                    (
                        contract["max_input_tokens"],
                        contract["max_new_tokens"],
                        contract["runtime_context_limit"],
                    ),
                    values,
                )
                assert_generation_request(contract, values[0])

        medical = build_generation_contract(profile=E2_EVALUATION_PROFILE)
        with self.assertRaises(ValueError):
            assert_generation_request(medical, 36865)
        drifted = dict(medical)
        drifted["max_input_tokens"] = 36865
        with self.assertRaises(ValueError):
            assert_strict_generation_contract(drifted)
        for key, value in (
            ("top_k", False),
            ("max_input_tokens", 36864.0),
            ("temperature", float("nan")),
        ):
            malformed = dict(medical)
            malformed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                assert_strict_generation_contract(malformed)
        unregistered = dict(medical)
        unregistered["repetition_penalty"] = 1.1
        with self.assertRaises(ValueError):
            assert_strict_generation_contract(unregistered)
        medical["stop_token_ids"][0] = 7
        with self.assertRaises(ValueError):
            assert_strict_generation_contract(medical)
        self.assertEqual(
            build_generation_contract(profile=E2_EVALUATION_PROFILE)["stop_token_ids"],
            [151645, 151643],
        )

    def test_target_mapping_and_mask(self) -> None:
        self.assertEqual(build_target_text("cot", "answer"), "cot\n\nanswer")
        self.assertEqual(
            supervision_mask(
                5,
                2,
                4,
                token_ids=[151644, 10, 20, 151645, 151643],
            ),
            (0, 0, 1, 1, 0),
        )
        with self.assertRaises(ValueError):
            build_target_text("", "answer")
        with self.assertRaises(ValueError):
            supervision_mask(5, 2, 2, token_ids=[1, 2, 3, 4, 5])
        with self.assertRaises(ValueError):
            supervision_mask(
                4,
                1,
                3,
                token_ids=[10, 151646, 151655, 11],
            )
        with self.assertRaises(ValueError):
            supervision_mask(5, 2, 4)
        self.assertEqual(
            supervision_mask(5, 2, 4, allow_unverified_target_tokens=True),
            (0, 0, 1, 1, 0),
        )
        with self.assertRaises(ValueError):
            supervision_mask(
                4,
                1,
                3,
                token_ids=[10, 90, 91, 11],
                special_token_ids=[90, True],
            )

    def test_c0_artifact_binds_compatibility_inputs_and_asserts_readiness(self) -> None:
        rendered = render_non_thinking_prompt("What is 2+2?")
        token_ids = [151644, 42, 151645]
        artifact = build_c0_artifact(
            rendered_prompt=rendered,
            teacher_rendered_token_ids=token_ids,
            student_rendered_token_ids=token_ids,
            teacher_revision="a" * 40,
            student_revision="b" * 40,
            vocab_sha256=QWEN3_VOCAB_SHA256,
            vocab_size=QWEN3_VOCAB_SIZE,
            tokenizer_length=QWEN3_TOKENIZER_LENGTH,
            special_token_map=QWEN3_SPECIAL_TOKEN_MAP,
            generation_profile=E1_EVALUATION_PROFILE,
        )
        assert_c0_production_ready(artifact)
        self.assertEqual(len(artifact.artifact_sha256), 64)
        restored = C0Artifact.from_dict(artifact.to_dict())
        self.assertEqual(restored, artifact)
        self.assertEqual(restored.artifact_sha256, artifact.artifact_sha256)
        signed = artifact.to_dict(include_artifact_sha256=True)
        self.assertEqual(C0Artifact.from_dict(signed), artifact)

        tampered = dict(signed)
        tampered["vocab_size"] = QWEN3_VOCAB_SIZE - 1
        with self.assertRaises(ValueError):
            C0Artifact.from_dict(tampered)
        unknown = artifact.to_dict()
        unknown["floating_override"] = True
        with self.assertRaises(ValueError):
            C0Artifact.from_dict(unknown)

        with self.assertRaises(ValueError):
            replace(artifact, serializer_version="floating-template").assert_production_ready()
        with self.assertRaises(ValueError):
            replace(artifact, vocab_sha256="0" * 64).assert_production_ready()
        with self.assertRaises(ValueError):
            replace(
                artifact, student_rendered_token_ids=(151644, 43, 151645)
            ).assert_production_ready()
        with self.assertRaises(ValueError):
            replace(
                artifact,
                special_token_map={**QWEN3_SPECIAL_TOKEN_MAP, "pad": 151642},
            ).assert_production_ready()
        with self.assertRaises(ValueError):
            C0Artifact.from_dict({})
        with self.assertRaises(ValueError):
            replace(artifact, teacher_revision="teacher-commit")

    def test_medical_system_prompt_is_explicit(self) -> None:
        rendered = render_non_thinking_prompt("What should I do?", MEDICAL_SYSTEM_PROMPT)
        self.assertIn("medical assistant", rendered.text)
        self.assertNotIn("<think>", rendered.text)


if __name__ == "__main__":
    unittest.main()
