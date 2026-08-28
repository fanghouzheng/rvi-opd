import unittest

from rvi_opd.context import assert_detached_contract, assert_rollback_contract, context_hash


class ContextContractTests(unittest.TestCase):
    def test_hash_is_sensitive_to_tokens_and_tokenizer(self) -> None:
        self.assertNotEqual(context_hash([1, 2], "a"), context_hash([1, 3], "a"))
        self.assertNotEqual(context_hash([1, 2], "a"), context_hash([1, 2], "b"))

    def test_detached_contract(self) -> None:
        assert_detached_contract([1, 2], [1, 2], [1, 2, 9], "tok")
        with self.assertRaises(AssertionError):
            assert_detached_contract([1, 2], [1, 2, 9], [1, 2, 9], "tok")

    def test_rollback_contract(self) -> None:
        assert_rollback_contract([1, 2], [1, 2], "tok")
        with self.assertRaises(AssertionError):
            assert_rollback_contract([1, 2], [1, 3], "tok")


if __name__ == "__main__":
    unittest.main()
