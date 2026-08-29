import unittest

from rvi_opd.tokenization import build_lexicon_artifact


class TokenizationTests(unittest.TestCase):
    def test_artifact_records_variants_sequences_and_deduplicated_ids(self) -> None:
        table = {
            "Wait": [10],
            " Wait": [20],
            "wait": [10],
            " wait": [20, 99],
        }
        artifact = build_lexicon_artifact(
            "relay",
            ["Wait"],
            table.__getitem__,
            tokenizer_revision="abc",
            tokenizer_sha256="tokenizer-hash",
            vocabulary_sha256="def",
            include_lowercase=True,
        )
        self.assertEqual(artifact.tokenizer_sha256, "tokenizer-hash")
        self.assertEqual(artifact.first_subword_ids, (10, 20))
        self.assertEqual(artifact.token_sequences[" wait"], (20, 99))
        self.assertEqual(len(artifact.artifact_sha256), 64)

    def test_empty_tokenization_fails(self) -> None:
        with self.assertRaises(ValueError):
            build_lexicon_artifact(
                "bad",
                ["Wait"],
                lambda _: [],
                "rev",
                "tokenizer-hash",
                "vocab",
                include_lowercase=False,
            )

    def test_empty_lexicon_fails(self) -> None:
        with self.assertRaises(ValueError):
            build_lexicon_artifact(
                "bad",
                [],
                lambda _: [1],
                "rev",
                "tokenizer-hash",
                "vocab",
                include_lowercase=False,
            )

    def test_relay_single_token_mode_drops_multitoken_variants(self) -> None:
        table = {
            "Wait": [10],
            " Wait": [20],
            "wait": [10],
            " wait": [20, 99],
        }
        artifact = build_lexicon_artifact(
            "relay",
            ["Wait"],
            table.__getitem__,
            tokenizer_revision="abc",
            tokenizer_sha256="tokenizer-hash",
            vocabulary_sha256="def",
            include_lowercase=True,
            single_token_only=True,
        )
        self.assertTrue(artifact.single_token_only)
        self.assertNotIn(" wait", artifact.token_sequences)
        self.assertEqual(artifact.first_subword_ids, (10, 20))

    def test_single_token_mode_fails_when_every_variant_is_multitoken(self) -> None:
        with self.assertRaises(ValueError):
            build_lexicon_artifact(
                "relay",
                ["Wait"],
                lambda _: [1, 2],
                "rev",
                "tokenizer-hash",
                "vocab",
                include_lowercase=False,
                single_token_only=True,
            )


if __name__ == "__main__":
    unittest.main()
