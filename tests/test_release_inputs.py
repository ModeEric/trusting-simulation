import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from scripts.acquire_inputs import acquire, destination


class ReleaseInputTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.data = b'archived result\n'
        self.record = dict(path='data/raw/fixture.csv', url='https://example.invalid/input',
                           bytes=len(self.data), sha256=hashlib.sha256(self.data).hexdigest(), bundled=False)

    def test_archive_restores_only_exact_bytes(self):
        archive = self.root / 'archive.zip'
        with ZipFile(archive, 'w') as bundle:
            bundle.writestr('trusting-simulation/' + self.record['path'], self.data)
        self.assertEqual(acquire(self.record, self.root, archive), 'acquired')
        self.assertEqual(acquire(self.record, self.root, verify_only=True), 'verified')

    def test_changed_download_is_never_written(self):
        with patch('scripts.acquire_inputs.urllib.request.urlopen') as response:
            response.return_value.__enter__.return_value.read.return_value = b'changed'
            with self.assertRaisesRegex(ValueError, 'differs'):
                acquire(self.record, self.root)
        self.assertFalse((self.root / self.record['path']).exists())

    def test_modified_local_input_is_preserved(self):
        path = self.root / self.record['path']
        path.parent.mkdir(parents=True)
        path.write_bytes(b'modified')
        with self.assertRaisesRegex(ValueError, 'preserved'):
            acquire(self.record, self.root)
        self.assertEqual(path.read_bytes(), b'modified')

    def test_verify_only_never_fetches_missing_input(self):
        with patch('scripts.acquire_inputs.urllib.request.urlopen') as request:
            with self.assertRaisesRegex(ValueError, 'Missing input'):
                acquire(self.record, self.root, verify_only=True)
            request.assert_not_called()

    def test_rejects_paths_outside_raw_directory(self):
        for name in ['data/raw/../../secrets', '/tmp/secrets', 'src/common.py']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                destination(self.root, name)

    def test_missing_bundled_input_requires_restoring_checkout(self):
        record = dict(self.record, bundled=True)
        with self.assertRaisesRegex(ValueError, 'restore the release checkout'):
            acquire(record, self.root)
