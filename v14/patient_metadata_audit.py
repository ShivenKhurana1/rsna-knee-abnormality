"""Read-only patient-tag audit. No fitting, pixel decoding, or identity reconstruction.

One first and one last DICOM header per series are checked; this is not an audit
of every slice. PatientID-backed grouping is conditional on the anonymizer having
preserved identity across examinations. That semantic guarantee is not inferred.
"""
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pydicom

UID = 'StudyInstanceUID'
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
PLACEHOLDERS = {'', 'anonymous', 'anonymized', 'anon', 'unknown', 'none', 'null', '0', '1'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_study(uid, series_ids, images_root, salt):
    tokens, errors, checked = set(), [], 0
    for sid in sorted(series_ids):
        files = sorted((images_root / uid / sid).glob('*.dcm'))
        if not files:
            errors.append('series_has_no_dicom')
            continue
        for file in sorted(set([files[0], files[-1]])):
            try:
                ds = pydicom.dcmread(file, stop_before_pixels=True,
                                    specific_tags=['PatientID', UID, 'SeriesInstanceUID'])
                checked += 1
                pid = str(ds.get('PatientID', '')).strip()
                if pid.lower() in PLACEHOLDERS:
                    errors.append('missing_or_placeholder_patient_id')
                elif pid == uid or pid == sid:
                    errors.append('patient_id_is_study_or_series_id')
                else:
                    # Equal IDs merge conservatively even across institutions.
                    # Raw patient identifiers and the random salt are never exported.
                    tokens.add(hashlib.sha256(salt + pid.encode()).hexdigest())
                if str(ds.get(UID, '')) != uid or str(ds.get('SeriesInstanceUID', '')) != sid:
                    errors.append('dicom_path_uid_mismatch')
            except Exception as exc:
                errors.append('header_read_error:' + type(exc).__name__)
    if len(tokens) != 1:
        errors.append('inconsistent_or_absent_patient_id_within_study')
    return {'StudyInstanceUID': uid, 'GroupID': next(iter(tokens)) if len(tokens) == 1 else '',
            'headers_checked': checked, 'series_checked': len(series_ids),
            'issues': sorted(set(errors))}


def audit(root, output, workers=12):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    train = pd.read_csv(root / 'train.csv', dtype={UID: str})
    series = pd.read_csv(root / 'train_series.csv', dtype={UID: str, 'SeriesInstanceUID': str})
    if train[UID].isna().any() or train[UID].duplicated().any():
        raise ValueError('Invalid study IDs')
    if set(train[UID]) != set(series[UID]) or series['SeriesInstanceUID'].duplicated().any():
        raise ValueError('Training/series coverage or uniqueness mismatch')
    grouped = series.groupby(UID)['SeriesInstanceUID'].agg(list).to_dict()
    salt, records = os.urandom(32), []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(inspect_study, uid, grouped[uid], root / 'train_series', salt): uid
                   for uid in train[UID]}
        for future in as_completed(futures):
            records.append(future.result())
            if len(records) % 250 == 0 or len(records) == len(train):
                print(f'PATIENT AUDIT {len(records)}/{len(train)} studies', flush=True)
    records.sort(key=lambda r: r[UID])
    valid = [r for r in records if not r['issues']]
    mapping = pd.DataFrame([{UID: r[UID], 'GroupID': r['GroupID']} for r in valid], columns=[UID, 'GroupID'])
    mapping.to_csv(output / 'patient_tag_groups.csv', index=False)
    group_sizes = mapping.groupby('GroupID').size()
    counts = {}
    for record in records:
        for issue in record['issues']:
            counts[issue] = counts.get(issue, 0) + 1
    gold = train[train[TARGETS].notna().all(axis=1)]
    gold_map = gold[[UID] + TARGETS].merge(mapping, on=UID, how='left', validate='one_to_one')
    eligible = len(valid) == len(train) and len(group_sizes) >= 20
    report = {
        'status': 'PATIENT_TAG_COVERAGE_COMPLETE_SEMANTICS_UNVERIFIED' if eligible else 'PATIENT_GROUPING_FAILED',
        'scope': 'First and last header in every series; no pixel or report identity reconstruction.',
        'studies': len(train), 'series': len(series), 'headers_checked': sum(r['headers_checked'] for r in records),
        'studies_with_usable_consistent_tags': len(valid), 'patient_tag_groups': len(group_sizes),
        'groups_with_multiple_studies': int((group_sizes > 1).sum()),
        'largest_group_studies': int(group_sizes.max()) if len(group_sizes) else 0,
        'issue_study_counts': counts, 'gold_studies': len(gold),
        'gold_patient_tag_groups': int(gold_map['GroupID'].nunique()),
        'missing_gold_group_ids': int(gold_map['GroupID'].isna().sum()),
        'input_sha256': {name: digest(root / name) for name in ['train.csv', 'train_series.csv']},
        'patient_map_sha256': digest(output / 'patient_tag_groups.csv'),
        'tag_coverage_gate_passed': eligible,
        'patient_identity_preservation_independently_verified': False,
        'baseline_v13_exclusion_verified': False,
        'training_performed': False, 'plus_0_02_verified': False,
        'caveat': 'Repeated examinations may have different anonymized PatientIDs. Header consistency alone cannot prove patient independence. Need host/source confirmation or an authoritative mapping.',
    }
    (output / 'patient_metadata_audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (output / 'private_study_audit.json').write_text(json.dumps(records) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.output)
