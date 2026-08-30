"""Run regression suites and save the real exit codes; no accuracy assertion."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from diagnostics import digest, write_new

ROOT = Path(__file__).resolve().parent.parent


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    commands = [
        ['-m', 'unittest', 'discover', '-s', 'v14', '-p', 'test_*.py', '-v'],
        ['v13/test_v13.py'], ['v13/test_comparison.py'],
    ]
    checks = []
    for command in commands:
        r = subprocess.run([sys.executable] + command, cwd=ROOT, capture_output=True, text=True)
        checks.append({'command': command, 'exit_code': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr})
        print('PASS' if r.returncode == 0 else 'FAIL', ' '.join(command), flush=True)
    report = {'all_passed': all(c['exit_code'] == 0 for c in checks), 'checks': checks,
              'scope': 'Regression tests; no real model accuracy assertion.',
              'v14_source_sha256': {f.name: digest(f) for f in sorted((ROOT / 'v14').glob('*.py'))}}
    write_new(a.output, report)
    sys.exit(0 if report['all_passed'] else 1)
