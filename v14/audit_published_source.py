"""Compare authenticated rendered V13 code cells with the frozen local source."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def audit():
    capture = HERE / 'private_artifacts/published_v13_code_cells.json'
    data = json.loads(capture.read_text(encoding='utf-8'))
    local = json.loads((HERE.parent / 'v13/rsna-knee-ensemble-v13.ipynb').read_text(encoding='utf-8'))
    a = [''.join(c['source']) for c in local['cells'] if c['cell_type'] == 'code']
    b = data['code_cells']
    if len(a) != len(b):
        raise ValueError('Missing or extra published code cells')
    changed = []
    for i, (x, y) in enumerate(zip(a, b)):
        compile(y, f'published-v13-{i}', 'exec')
        if ast.dump(ast.parse(x)) != ast.dump(ast.parse(y)):
            changed.append({'code_cell_index': i, 'diff': ''.join(difflib.unified_diff(
                x.splitlines(True), y.splitlines(True), fromfile='local', tofile='published'))})
    result = {'source': data['source'], 'code_cells': len(a), 'ast_identical_cells': len(a) - len(changed),
              'differences': changed, 'capture_sha256': hashlib.sha256(capture.read_bytes()).hexdigest(),
              'complete_notebook_byte_parity_verified': False,
              'conclusion': 'Published V13 already uses %.17g CSV formatting and equal_nan=True. Local V13 does not. All other rendered code cells match by AST.',
              'validation_adjustment': 'V14 uses %.17g, retaining stricter NaN rejection; inference predictions must be finite.'}
    (HERE / 'published_v13_source_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    audit()
