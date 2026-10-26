import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from src import reproduce


class ReproductionStatusTests(unittest.TestCase):
    def test_checksum_failure_replaces_stale_success(self):
        with tempfile.TemporaryDirectory() as folder:
            destination=Path(folder)
            (destination/'validation.json').write_text('{"status":"passed"}')
            with patch.object(reproduce,'RESULTS',destination), patch.object(reproduce,'verify_raw',side_effect=ValueError('checksum mismatch')):
                with self.assertRaises(ValueError):reproduce.main()
            state=json.loads((destination/'validation.json').read_text())
            self.assertEqual(state['status'],'failed')
            self.assertEqual(state['failure_stage'],'input_verification')
