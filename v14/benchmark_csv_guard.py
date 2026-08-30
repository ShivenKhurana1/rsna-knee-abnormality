"""Reproduce local CSV guard failure; this is a numeric test, NOT MRI accuracy."""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd


def benchmark():
    values = np.random.default_rng(14).uniform(size=(58, 12)).astype(np.float32)
    frame = pd.DataFrame(values)
    report = {'scope': 'Synthetic float32 serialization test, not knee-model accuracy',
              'shape': list(values.shape), 'seed': 14, 'dtype': str(values.dtype), 'formats': {}}
    for name, fmt in [('local_v13_default', None), ('published_v13_and_corrected_v14', '%.17g')]:
        csv = frame.to_csv(index=False, float_format=fmt)
        restored = pd.read_csv(io.StringIO(csv)).to_numpy(float)
        delta = np.abs(restored - values.astype(float))
        report['formats'][name] = {'max_absolute_error': float(delta.max()),
                                  'cells_above_guard_tolerance': int((delta > (1e-12 + 1e-12 * np.abs(values))).sum()),
                                  'passes_original_strict_guard': bool(np.allclose(restored, values.astype(float), atol=1e-12, rtol=1e-12)),
                                  'float32_values_recovered_exactly': bool(np.array_equal(restored.astype(np.float32), values))}
    path = Path(__file__).with_name('csv_guard_benchmark.json')
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    benchmark()
