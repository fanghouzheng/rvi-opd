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


if __name__ == "__main__":
    unittest.main()
