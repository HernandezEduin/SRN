"""Progress reporting tests that do not require TensorFlow or dataset downloads."""

import contextlib
import io
import unittest

from theseus_experiment import ProgressRecords, ProgressReporter, format_duration


class ProgressTests(unittest.TestCase):
    def test_duration(self):
        self.assertEqual(format_duration(0), "00:00:00")
        self.assertEqual(format_duration(3661.9), "01:01:01")

    def test_evaluation_wrapper_preserves_records(self):
        records = [("q1",), ("q2",)]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with ProgressReporter("test", len(records), 0) as progress:
                wrapped = ProgressRecords(records, progress)
                self.assertEqual(len(wrapped), len(records))
                self.assertEqual(list(wrapped), records)
        self.assertIn("test: 2/2 (100.0%)", output.getvalue())

    def test_reporter_handles_no_completed_queries(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with ProgressReporter("warmup", 10, 0) as progress:
                progress.report()
        self.assertIn("ETA estimating", output.getvalue())


if __name__ == "__main__":
    unittest.main()
