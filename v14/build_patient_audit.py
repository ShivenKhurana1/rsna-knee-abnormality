"""Build a private CPU-only pre-training audit notebook; no model training."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build():
    source = (HERE / 'patient_metadata_audit.py').read_text(encoding='utf-8')
    code = 'import types\npatient_audit = types.ModuleType("patient_audit")\nexec(' + repr(source) + ', patient_audit.__dict__)\n'
    code += '''from pathlib import Path
candidates = [Path('/kaggle/input/rsna-knee-abnormality-detection'),
              Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')]
roots = [p for p in candidates if (p / 'train.csv').exists()]
if len(roots) != 1:
    raise RuntimeError('Expected exactly one competition root')
PATIENT_AUDIT = patient_audit.audit(roots[0], '/kaggle/working/patient_audit', workers=12)
print('NO TRAINING OR SUBMISSION. Patient identity semantics and V13 exclusions remain unverified.')
'''
    compile(code, '<patient-audit>', 'exec')
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                'cells': [
                    {'cell_type': 'markdown', 'id': 'patient-audit-intro', 'metadata': {}, 'source': [
                        '# V14 patient-tag audit — private, CPU only\n',
                        'Check all training studies before new specialist training. No model fitting, pixel decoding, report identity reconstruction, or leaderboard submission.\n',
                        'PatientID consistency is not proof the anonymizer preserved patient identity between examinations. Raw patient IDs are never exported.\n']},
                    {'cell_type': 'code', 'id': 'patient-audit-run', 'metadata': {},
                     'execution_count': None, 'outputs': [], 'source': code.splitlines(keepends=True)}]}
    target = HERE / 'rsna-knee-v14-patient-audit.ipynb'
    target.write_text(json.dumps(notebook, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({'notebook': str(target), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                      'training_enabled': False, 'gpu_required': False}, indent=2))
    return notebook


if __name__ == '__main__':
    build()
