"""Build the private clean-reference training pilot, with no embedded patient data."""
import hashlib
import json
from pathlib import Path

from clean_specialist import PROTOCOL

HERE = Path(__file__).resolve().parent


def build():
    modules = ['diagnostics', 'clean_specialist', 'clean_features']
    source = 'import sys, types, os\nos.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"\nos.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"\n'
    hashes = {}
    for name in modules:
        code = (HERE/f'{name}.py').read_text(encoding='utf-8')
        hashes[name] = hashlib.sha256(code.encode()).hexdigest()
        source += f'{name} = types.ModuleType({name!r})\nsys.modules[{name!r}] = {name}\nexec({code!r}, {name}.__dict__)\n'
    run = '''from pathlib import Path
import json, time, zipfile
import pandas as pd
import numpy as np
from threadpoolctl import threadpool_limits
from clean_specialist import PROTOCOL, TARGETS, sha, splits
start = time.perf_counter()
output = Path('/kaggle/working/clean_specialist')
output.mkdir(exist_ok=False)
root_candidates = [Path('/kaggle/input/rsna-knee-abnormality-detection'),
                   Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')]
roots = [p for p in root_candidates if (p/'train.csv').exists() and (p/'train_series.csv').exists()]
if len(roots) != 1:
    raise RuntimeError('Attach only the competition plus patient-audit notebook output')
map_candidates = list(Path('/kaggle/input').glob('*/patient_audit/patient_tag_groups.csv'))
map_candidates += list(Path('/kaggle/input').glob('notebooks/*/*/patient_audit/patient_tag_groups.csv'))
maps = [p for p in map_candidates
        if sha(p) == '83a1c3945988d83241319678dc815828e3d1fbb2b33b1276c2b3d38ecf091df4']
if len(maps) != 1:
    raise RuntimeError('Attach RSNA Knee V14 Patient Audit version 346132000 output')
mapping = pd.read_csv(maps[0], dtype=str)
if mapping.StudyInstanceUID.duplicated().any() or mapping.isna().any().any():
    raise ValueError('Invalid PatientID grouping map')
train = pd.read_csv(roots[0]/'train.csv', dtype={'StudyInstanceUID': str})
ids = train.loc[train[TARGETS].notna().all(axis=1), 'StudyInstanceUID'].sort_values().to_numpy(str)
groups = mapping.set_index('StudyInstanceUID').loc[ids, 'GroupID'].to_dict()
# Freeze this recipe and partitions before model download, image extraction or fits.
assignment = np.full(len(ids), -1)
for k, (_, va) in enumerate(splits(np.array([groups[u] for u in ids]), PROTOCOL['outer_folds'], PROTOCOL['seed'])):
    assignment[va] = k
pd.DataFrame({'StudyInstanceUID': ids, 'GroupID': [groups[u] for u in ids], 'fold': assignment}).to_csv(output/'predeclared_folds.csv', index=False)
(output/'predeclared_protocol.json').write_text(json.dumps(PROTOCOL, indent=2)+'\\n')
print('PROVISIONAL CLEAN REFERENCE: no V13 weights, no weak/report labels, no leaderboard submission', flush=True)
features = clean_features.extract(roots[0], groups, output/'features')
with threadpool_limits(limits=2):
    results = clean_specialist.evaluate_features(features, output/'evaluation')
runtime = {'wall_seconds': time.perf_counter()-start, 'new_training_completed': True,
           'patient_independence_confirmed': False, 'v13_gain_verified': False,
           'source_sha256': SOURCE_HASHES, 'recipe_sha256': sha(output/'predeclared_protocol.json')}
(output/'run_receipt.json').write_text(json.dumps(runtime, indent=2)+'\\n')
with zipfile.ZipFile('/kaggle/working/clean_specialist_artifacts.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for file in sorted(output.rglob('*')):
        if file.is_file():
            archive.write(file, file.relative_to(output))
print('CLEAN SPECIALIST TRAINING AND BENCHMARK COMPLETE', json.dumps(runtime), flush=True)
'''
    run = 'SOURCE_HASHES = '+repr(hashes)+'\n'+run
    for text in [source, run]:
        compile(text, '<clean-specialist>', 'exec')
    cells = [{'cell_type': 'markdown', 'id': 'clean-intro', 'metadata': {}, 'source': [
        '# Clean-reference regional specialist pilot — private research\n',
        'New regularized heads, frozen external DINOv2-small, five outer/three inner PatientID-grouped folds.\n',
        '**Not exact V13. Not confirmed patient independence. Historical 58-case gold is exploratory.**\n',
        'Attach the competition and RSNA Knee V14 Patient Audit output (346132000), enable a GPU and internet for the pinned public encoder. No leaderboard submission.\n']},
        *[{'cell_type': 'code', 'id': f'clean-code-{i}', 'metadata': {}, 'execution_count': None,
           'outputs': [], 'source': code.splitlines(keepends=True)} for i, code in enumerate([source, run])]]
    nb = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'cells': cells}
    path = HERE/'rsna-knee-v14-clean-specialist.ipynb'
    path.write_text(json.dumps(nb, indent=1)+'\n', encoding='utf-8')
    manifest = {'protocol': PROTOCOL, 'source_sha256': hashes,
                'notebook_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'executed': False}
    (HERE/'clean_specialist_build.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'notebook': str(path), 'sha256': manifest['notebook_sha256']}, indent=2))
    return nb


if __name__ == '__main__':
    build()
