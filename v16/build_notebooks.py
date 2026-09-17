"""Build self-contained offline notebooks. No code upload or submission."""
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
MODULES = ('__init__.py', '__main__.py', 'common.py', 'data.py', 'model.py',
           'validation.py', 'runner.py', 'evaluation.py')


def code(source):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': source.splitlines(keepends=True)}


def markdown(source):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': source.splitlines(keepends=True)}


def build():
    files = {name: (HERE/name).read_text(encoding='utf-8') for name in MODULES+('config.json',)}
    embed = "import os, sys, json\nfrom pathlib import Path\n"
    embed += "os.environ['HF_HUB_OFFLINE']='1'\nos.environ['TRANSFORMERS_OFFLINE']='1'\n"
    embed += "runtime=Path('/kaggle/working/v16_runtime')\n(runtime/'v16').mkdir(parents=True, exist_ok=True)\n"
    for name, source in files.items():
        embed += f"(runtime/'v16'/{name!r}).write_text({source!r}, encoding='utf-8')\n"
    embed += "sys.path.insert(0, str(runtime))\n"
    settings = '''# Edit these paths to match your attached data/models; all are runtime settings.
DATA_ROOT = os.environ.get('RSNA_DATA_ROOT', '')
WORK = os.environ.get('V16_WORK', '/kaggle/working/v16_work')
ENCODER_PATH = os.environ.get('V16_ENCODER_PATH', '')
DINOV2_REPO = os.environ.get('V16_DINOV2_REPO', '')
REPORT_LABELS = os.environ.get('V16_REPORT_LABELS', '')
GROUPS_CSV = os.environ.get('V16_GROUPS_CSV', '')
FOLD = int(os.environ.get('V16_FOLD', '0'))
SEED = int(os.environ.get('V16_SEED', '1400'))
ARM = os.environ.get('V16_ARM', 'platt')
CHECKPOINTS = json.loads(os.environ.get('V16_CHECKPOINTS', '[]'))

if not DATA_ROOT:
    matches = []
    for folder, dirs, names in os.walk('/kaggle/input'):
        if 'sample_submission.csv' in names and ('test.csv' in names or 'train.csv' in names):
            matches.append(folder)
        dirs[:] = [d for d in dirs if d not in ('train_series', 'test_series')]
    if len(matches) != 1:
        raise ValueError('Set RSNA_DATA_ROOT to the intended competition mount')
    DATA_ROOT = matches[0]
print('Data root:', DATA_ROOT)
'''
    prepare_code = '''from v16.common import config
from v16.runner import prepare
cfg = config(runtime/'v16/config.json')
cfg.update(encoder_path=ENCODER_PATH, dinov2_repo=DINOV2_REPO, auxiliary=ARM)
if not ENCODER_PATH:
    raise ValueError('Set V16_ENCODER_PATH to the attached DINOv2 HF directory or native .pth')
receipt = prepare(DATA_ROOT, WORK, cfg, REPORT_LABELS or None, GROUPS_CSV or None)
print(json.dumps(receipt, indent=2))
'''
    train_code = '''# WORK must be writable. Copy prepared artifacts to local scratch before training.
from v16.runner import train
receipt = train(WORK, FOLD, seed=SEED, arm=ARM)
print(json.dumps(receipt, indent=2))
'''
    inference_code = '''from v16.runner import infer
if not CHECKPOINTS:
    raise ValueError('Set V16_CHECKPOINTS to a JSON list of attached trained V16 checkpoint paths')
receipt = infer(DATA_ROOT, CHECKPOINTS, '/kaggle/working/submission.csv',
                '/kaggle/working/v16_test_cache', device='cuda',
                encoder_code=DINOV2_REPO or None)
print(json.dumps(receipt, indent=2))
'''
    inventory = {}
    for name, body, purpose in [('prepare', prepare_code, 'Prepare labels, group folds, and real DICOM caches. Do not submit.'),
                                ('train', train_code, 'Train one matched fold/seed/supervision arm. Do not submit.'),
                                ('infer', inference_code, 'Predict the live test set from trained V16 checkpoints. No score is claimed.')]:
        nb = {'nbformat': 4, 'nbformat_minor': 5,
              'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
              'cells': [markdown(f'# RSNA Knee V16 — {name}\n\n{purpose}\n\n'
                                 'Requires attached offline dependencies and the inputs described in README.md. '
                                 'This notebook performs no internet downloads.'), code(embed), code(settings), code(body)]}
        for i, cell in enumerate(nb['cells']):
            cell['id'] = f'v16-{name}-{i}'
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), cell['id'], 'exec')
        path = HERE/f'rsna-knee-v16-{name}.ipynb'
        path.write_text(json.dumps(nb, indent=1), encoding='utf-8')
        inventory[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {'notebooks': inventory, 'embedded_modules': {k: hashlib.sha256(v.encode()).hexdigest() for k, v in files.items()},
                'scored_auc': None, 'uploaded': False}
    (HERE/'build_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    package = HERE/'rsna-knee-v16.zip'
    with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(HERE.iterdir()):
            if path.suffix in ('.py', '.json', '.md', '.txt', '.ipynb'):
                archive.write(path, 'v16/'+path.name)
    print(json.dumps({'package': str(package), **manifest}, indent=2))


if __name__ == '__main__':
    build()
