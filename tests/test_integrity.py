import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('reproduce',ROOT/'reproduce.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
class IntegrityTests(unittest.TestCase):
 def test_real_distribution_is_complete(self):self.assertGreater(r.check_inputs(),40)
 def test_missing_and_tampered_data_fail(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);(root/'data').mkdir();(root/'reference').mkdir()
   (root/'data/integrity_manifest.json').write_text(json.dumps({'data/required.csv':'0'*64}))
   with patch.object(r,'ROOT',root):
    with self.assertRaisesRegex(RuntimeError,'Missing or changed'):r.check_inputs()
    (root/'data/required.csv').write_text('wrong data')
    with self.assertRaisesRegex(RuntimeError,'Missing or changed'):r.check_inputs()
 def test_corrupt_prediction_archive_fails(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);bad=root/'predictions.zip';bad.write_bytes(b'not a zip')
   (root/'release_manifest.json').write_text(json.dumps({'asset':'predictions.zip','sha256':'0'*64}))
   with patch.object(r,'ROOT',root):
    with self.assertRaisesRegex(RuntimeError,'checksum failed'):r.predictions(root,bad)
 def test_output_cannot_recurse_into_source_tools(self):
  target=ROOT/'tools/forbidden_output_regression'
  with patch('sys.argv',['reproduce.py','--mode','smoke','--out',str(target)]):
   with self.assertRaisesRegex(ValueError,'outside frozen inputs'):r.main()
  self.assertFalse(target.exists())
if __name__=='__main__':unittest.main()
