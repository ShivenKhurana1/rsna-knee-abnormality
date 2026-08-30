"""Compare exact numeric CSV cells captured from the authenticated Kaggle UI.

This compares the full displayed CSV tables, NOT downloaded original file bytes.
No labels are present, so this script deliberately does not compute AUC.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from diagnostics import UID, TARGETS, digest, prediction_delta, write_new


def load_table(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    rows = [r for r in data['rows'] if r]
    if rows[0] != [UID] + TARGETS or any(len(row) != 13 for row in rows):
        raise ValueError('Unexpected or truncated Kaggle table')
    frame = pd.DataFrame(rows[1:], columns=rows[0])
    frame[TARGETS] = frame[TARGETS].astype(float)
    return frame, data['source']


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v11', type=Path, required=True)
    p.add_argument('--v13', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    b, bu = load_table(a.v11)
    c, cu = load_table(a.v13)
    result = prediction_delta(b, c)
    result.update(sources={'v11': bu, 'v13': cu},
                  extraction_sha256={'v11': digest(a.v11), 'v13': digest(a.v13)},
                  original_csv_byte_equality_verified=False,
                  scope='All displayed numeric CSV cells; 3 visible test studies only, not hidden scoring rows.',
                  accuracy_change_measured=False)
    write_new(a.output, result)
    print(json.dumps(result, indent=2))
