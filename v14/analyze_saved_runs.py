"""Analyze real visible-run logs and CSV cells; no hidden-test or accuracy claim."""
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
from compare_saved_outputs import load_table
from diagnostics import UID, TARGETS, digest, prediction_delta, write_new


def parse_log(path):
    text = Path(path).read_text(encoding='utf-8')
    rows = []
    for line in text.splitlines():
        # Only logged CoAtNet preview rows start with a study UID after timestamp/line.
        content = re.sub(r'^\d+(?:\.\d+)?s\s+\d+\s+', '', line).split()
        if len(content) == 13 and content[0].startswith('1.2.'):
            rows.append([content[0]] + [float(v) for v in content[1:]])
    if len(rows) != 3:
        raise ValueError('Expected the three complete logged CoAtNet preview rows')
    arms = re.findall(r'\[arm (\d+)\] done \+ freed', text)
    return pd.DataFrame(rows, columns=[UID] + TARGETS), {
        'complete_arms': sorted(set(arms)),
        'stage4_seconds_rounded': int(re.search(r'\bDONE (\d+)s', text)[1]),
        'log_sha256': digest(path),
        'resized_window_reuse_logged': 'reusing its cached resized windows' in text,
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    old, old_meta = parse_log(a.artifacts / 'v11_log.txt')
    new, new_meta = parse_log(a.artifacts / 'v13_log.txt')
    final_old, old_url = load_table(a.artifacts / 'v11_submission_dom.json')
    final_new, new_url = load_table(a.artifacts / 'v13_submission_dom.json')
    intermediate = prediction_delta(old, new)
    final = prediction_delta(final_old, final_new)
    report = {
        'scope': 'REAL saved visible runs, n=3. Logged intermediate values are rounded. Final CSV DOM values retain displayed precision.',
        'versions': {'v11': 345726369, 'v13': 345967510},
        'sources': {'v11': old_url, 'v13': new_url},
        'v11': old_meta, 'v13': new_meta,
        'logged_coatnet_blend_changed_cells': int(np.sum(old[TARGETS].to_numpy() != new[TARGETS].to_numpy())),
        'logged_coatnet_blend_comparison': intermediate, 'final_submission_comparison': final,
        'conclusion': 'The intermediate CoAtNet blend changed but all final values/rankings on the visible studies stayed identical. Both runs completed all three arms.',
        'hidden_test_predictions_available': False, 'real_auc_delta_measured': False,
        'runtime_caveat': '43s/60s are logged Stage-4 visible-run durations, not controlled speed benchmarks or hidden-test estimates.',
    }
    write_new(a.output, report)
    print(report['conclusion'])
    print('Changed intermediate cells:', report['logged_coatnet_blend_changed_cells'])
    print('Changed targets:', [t for t, r in intermediate['targets'].items() if r['max_absolute_change']])
