"""Read public Kaggle source/metadata without executing downloaded notebook code."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request


REFERENCES = [
    'anhadmahajan06/rsna-knee-take-care-of-your-knee',
    'anvithpothula/rsna-base',
    'prvsiyan/head-and-shoulders-knees-and-toes',
    'evgendvorkin/rsna-baseline',
    'romantamrazov/rsna-knee-dinosaur-v4',
    'romantamrazov/rsna-knee-dinosaur-v3-train',
    'dreaddevelopment/knee-mri-twelve-findings-from-a-single-model',
    'dreaddevelopment/knee-mri-training-the-twelve-finding-model',
]


def fetch_one(ref, destination):
    url = 'https://www.kaggle.com/api/v1/kernels/pull/' + ref
    with urllib.request.urlopen(url, timeout=45) as response:
        payload = json.load(response)
    metadata = payload['metadata']
    if metadata.get('isPrivate'):
        raise ValueError('Only public source is in scope')
    source = payload['blob']['source']
    name = ref.replace('/', '__')
    if metadata.get('kernelType') == 'notebook':
        notebook = json.loads(source)
        (destination / (name + '.ipynb')).write_text(source, encoding='utf-8')
        # A searchable, nonexecuted source view for the audit, not an inference script.
        code = '\n\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
        cell_count = len(notebook['cells'])
    else:
        code, cell_count = source, None
    (destination / (name + '.txt')).write_text(code, encoding='utf-8')
    row = {
        'ref': ref, 'url': 'https://www.kaggle.com/code/' + ref,
        'version': metadata.get('currentVersionNumber'),
        'last_run': metadata.get('lastRunTime'),
        'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'cells': cell_count,
        'datasets': metadata.get('datasetDataSources', []),
        'kernels': metadata.get('kernelDataSources', []),
        'models': metadata.get('modelDataSources', []),
    }
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--ref', action='append')
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda ref: fetch_one(ref, args.destination), args.ref or REFERENCES))
    (args.destination / 'source_inventory.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
