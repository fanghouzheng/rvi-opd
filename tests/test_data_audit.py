import unittest

from rvi_opd.data_audit import build_prompt_manifest, normalize_prompt


class DataAuditTests(unittest.TestCase):
    def test_normalization_and_duplicate_manifest(self) -> None:
        rows = [
            {"id": "a", "prompt": "x   + y"},
            {"id": "b", "prompt": "x + y\n"},
            {"id": "c", "prompt": "x - y"},
        ]
        manifest = build_prompt_manifest(rows, "id", "prompt")
        self.assertEqual(manifest["physical_rows"], 3)
        self.assertEqual(manifest["unique_normalized_prompts"], 2)
        self.assertEqual(manifest["duplicate_physical_rows"], 1)
        self.assertEqual(manifest["duplicate_groups"][0]["source_ids"], ["a", "b"])
        self.assertNotIn("prompt", manifest["records"][0])

    def test_nfkc(self) -> None:
        self.assertEqual(normalize_prompt("Ａ  B"), "A B")

    def test_duplicate_source_id_fails(self) -> None:
        with self.assertRaises(ValueError):
            build_prompt_manifest(
                [{"id": "a", "prompt": "x"}, {"id": "a", "prompt": "y"}],
                "id",
                "prompt",
            )


if __name__ == "__main__":
    unittest.main()
