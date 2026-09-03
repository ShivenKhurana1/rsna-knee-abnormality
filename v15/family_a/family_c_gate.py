"""Family C reproducibility gate: only proceed to Family C once Family A shows a
reproducible, same-direction gain across two independently seeded training runs.
This is a project-management gate, not statistical proof (the plan is explicit
about that distinction) -- with a 58-study cohort, requiring the bootstrap CI
lower bound to clear zero on both seeds would almost never pass regardless of
whether the underlying effect is real, so the gate checks sign- and
magnitude-consistency across seeds instead of a per-seed significance test.
"""
import argparse
import json
from pathlib import Path


def evaluate_gate(seed_reports, min_macro_gain=0.0):
    """seed_reports: list of >=2 compare_oof.py-style report dicts, one per
    training seed, each comparing the same two arms. PASS requires every seed's
    macro_delta to exceed min_macro_gain AND all seeds to agree in sign."""
    if len(seed_reports) < 2:
        raise ValueError('Need at least two seeds to evaluate reproducibility')
    deltas = [r['macro_delta'] for r in seed_reports]
    signs = {1 if d > 0 else (-1 if d < 0 else 0) for d in deltas}
    all_above_threshold = all(d > min_macro_gain for d in deltas)
    consistent_sign = len(signs) == 1 and 0 not in signs
    passed = all_above_threshold and consistent_sign

    per_target_agreement = {}
    targets = seed_reports[0]['targets'].keys()
    for t in targets:
        target_deltas = [r['targets'][t]['delta'] for r in seed_reports]
        target_signs = {1 if d > 0 else (-1 if d < 0 else 0) for d in target_deltas}
        per_target_agreement[t] = {
            'deltas_by_seed': target_deltas,
            'sign_agreement': len(target_signs) == 1 and 0 not in target_signs,
        }
    agreeing_targets = sum(v['sign_agreement'] for v in per_target_agreement.values())

    return {
        'decision': 'PASS_PROCEED_TO_FAMILY_C' if passed else 'FAIL_DO_NOT_START_FAMILY_C',
        'reason': ('all seeds exceed the minimum gain with consistent sign' if passed else
                   'macro deltas disagree in sign or direction across seeds' if not consistent_sign else
                   f'not every seed exceeded the minimum gain of {min_macro_gain}'),
        'macro_deltas_by_seed': deltas,
        'min_macro_gain_required': min_macro_gain,
        'targets_agreeing_in_sign': agreeing_targets, 'targets_total': len(per_target_agreement),
        'per_target': per_target_agreement,
        'caveat': ('Project-management gate, not statistical proof: 58 studies do not support a '
                  'per-seed significance requirement. A PASS means the observed direction was '
                  'reproduced, not that the effect is confirmed at conventional significance.'),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed-reports', type=Path, nargs='+', required=True,
                   help='Two or more compare_oof.py JSON outputs, one per training seed')
    p.add_argument('--min-macro-gain', type=float, default=0.0)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    reports = [json.loads(path.read_text(encoding='utf-8')) for path in args.seed_reports]
    result = evaluate_gate(reports, args.min_macro_gain)
    with args.out.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(result['decision'])
    print(result['reason'])
    print(f"targets agreeing in sign: {result['targets_agreeing_in_sign']}/{result['targets_total']}")


if __name__ == '__main__':
    main()
