"""Synthetic DICOM header tests; no patient/model accuracy claim."""
import tempfile
import unittest
from pathlib import Path

try:
    import pydicom
except ImportError:
    pydicom = None


@unittest.skipIf(pydicom is None, 'pydicom required for DICOM audit tests')
class PatientMetadataTests(unittest.TestCase):
    def header(self, root, study, series, instance, pid):
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        path = root / study / series / (instance + '.dcm')
        path.parent.mkdir(parents=True, exist_ok=True)
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0' * 128)
        ds.StudyInstanceUID = study
        ds.SeriesInstanceUID = series
        if pid is not None:
            ds.PatientID = pid
        ds.save_as(path, enforce_file_format=True)
        return path

    def test_repeated_patient_maps_to_same_group_not_study(self):
        from patient_metadata_audit import inspect_study
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.header(root, '1.2.3', '1.2.3.1', 'a', 'TEST_PATIENT_A')
            self.header(root, '1.2.4', '1.2.4.1', 'a', 'TEST_PATIENT_A')
            a = inspect_study('1.2.3', ['1.2.3.1'], root, b'test-salt')
            b = inspect_study('1.2.4', ['1.2.4.1'], root, b'test-salt')
            self.assertFalse(a['issues'])
            self.assertFalse(b['issues'])
            self.assertEqual(a['GroupID'], b['GroupID'])
            self.assertNotIn('TEST_PATIENT_A', str(a))

    def test_missing_placeholder_and_study_ids_are_rejected(self):
        from patient_metadata_audit import inspect_study
        for pid in [None, '', 'Anonymous', '1.2.3']:
            with self.subTest(pid=pid), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                self.header(root, '1.2.3', '1.2.3.1', 'a', pid)
                row = inspect_study('1.2.3', ['1.2.3.1'], root, b'test-salt')
                self.assertTrue(row['issues'])
                self.assertEqual(row['GroupID'], '')

    def test_inconsistent_first_last_patient_ids_rejected(self):
        from patient_metadata_audit import inspect_study
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.header(root, '1.2.3', '1.2.3.1', 'a', 'TEST_A')
            self.header(root, '1.2.3', '1.2.3.1', 'z', 'TEST_B')
            row = inspect_study('1.2.3', ['1.2.3.1'], root, b'test-salt')
            self.assertEqual(row['headers_checked'], 2)
            self.assertIn('inconsistent_or_absent_patient_id_within_study', row['issues'])

    def test_dicom_path_mismatch_rejected(self):
        from patient_metadata_audit import inspect_study
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = self.header(root, '1.2.3', '1.2.3.1', 'a', 'TEST_A')
            ds = pydicom.dcmread(path)
            ds.StudyInstanceUID = '1.2.9'
            ds.save_as(path, enforce_file_format=True)
            row = inspect_study('1.2.3', ['1.2.3.1'], root, b'test-salt')
            self.assertIn('dicom_path_uid_mismatch', row['issues'])


if __name__ == '__main__':
    unittest.main()
