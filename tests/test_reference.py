import itertools
import unittest
from unittest.mock import patch

from stochastic_provenance import __main__ as demo


class ReferenceTests(unittest.TestCase):
    def test_matches_expected_paths(self):
        self.assertEqual(
            demo.run(demo.PREFIX, demo.DRAWS, shared=True),
            (((0, 1, 0, 1), (1, 0, 1, 0), (1, 0, 1, 0)), 1),
        )

    def test_exhaustive_two_step_coupling(self):
        paths = tuple(itertools.product(range(16), repeat=2))
        for prefix in ((), (0,), (1, 0, 1)):
            with self.subTest(prefix=prefix):
                serial, _ = demo.run(prefix, paths, shared=False)
                shared, _ = demo.run(prefix, paths, shared=True)
                self.assertEqual(serial, shared)

    def test_prefill_counts_are_observed(self):
        for shared, count in ((False, 3), (True, 1)):
            with patch.object(demo, "prefill", wraps=demo.prefill) as prefill:
                _, reported = demo.run(demo.PREFIX, demo.DRAWS, shared=shared)
                self.assertEqual(reported, count)
                self.assertEqual(prefill.call_count, count)

    def test_path_order_does_not_change_sampling(self):
        original, _ = demo.run(demo.PREFIX, demo.DRAWS, shared=True)
        reordered, _ = demo.run(demo.PREFIX, reversed(demo.DRAWS), shared=True)
        self.assertEqual(original, tuple(reversed(reordered)))

    def test_negative_control_changes_path(self):
        normal, _ = demo.run(demo.PREFIX, demo.DRAWS, shared=False)
        perturbed, _ = demo.run(demo.PREFIX, demo.DRAWS, shared=True, perturb=True)
        self.assertNotEqual(normal, perturbed)

    def test_no_paths_has_no_prefill(self):
        self.assertEqual(demo.run(demo.PREFIX, (), shared=True), ((), 0))
        self.assertEqual(demo.run(demo.PREFIX, (), shared=False), ((), 0))

    def test_invalid_draw_rejected(self):
        for value in (-1, 16, 2.0, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                demo.continuation(demo.PREFIX, (value,))


if __name__ == "__main__":
    unittest.main()
