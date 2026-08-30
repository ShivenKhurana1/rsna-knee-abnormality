"""Deterministically generate a self-contained inference-only V11 from V10.

Edits are fail-closed: named functions and replacement anchors must be unique.
The V9 crop functions are copied exactly, not approximated. No notebook executes.
"""

import ast
import copy
import hashlib
import json
from pathlib import Path
import textwrap


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def source(cell):
    return ''.join(cell['source'])


def set_source(cell, code):
    cell['source'] = code.splitlines(keepends=True)


def replace_once(code, old, new):
    if code.count(old) != 1:
        raise ValueError(f'Expected one replacement anchor, got {code.count(old)}: {old[:90]!r}')
    return code.replace(old, new, 1)


def function_span(code, name):
    matches = [node for node in ast.walk(ast.parse(code))
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    if len(matches) != 1:
        raise ValueError(f'Expected one {name}, found {len(matches)}')
    node = matches[0]
    return min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1, node.end_lineno


def find_function(notebook, name):
    matches = []
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        code = source(cell)
        try:
            start, end = function_span(code, name)
        except ValueError:
            continue
        matches.append(textwrap.dedent(''.join(code.splitlines(keepends=True)[start:end])))
    if len(matches) != 1:
        raise ValueError(f'Notebook must contain exactly one {name}')
    return matches[0]


def replace_function(code, name, replacement):
    lines = code.splitlines(keepends=True)
    start, end = function_span(code, name)
    prefix = lines[start][:len(lines[start]) - len(lines[start].lstrip())]
    replacement = textwrap.indent(textwrap.dedent(replacement).strip() + '\n', prefix)
    return ''.join(lines[:start]) + replacement + ''.join(lines[end:])


def code_cell(code, identifier):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {},
            'outputs': [], 'id': identifier, 'source': code.splitlines(keepends=True)}


def build():
    v10_path = ROOT / 'v10' / 'rsna-knee-ensemble-v10.ipynb'
    v9_path = ROOT / 'v9' / 'rsna-knee-ensemble-v9.ipynb'
    original = json.loads(v10_path.read_text(encoding='utf-8'))
    v9 = json.loads(v9_path.read_text(encoding='utf-8'))
    notebook = copy.deepcopy(original)
    cells = notebook['cells']
    if len(cells) != 38:
        raise ValueError('V10 cell layout changed; review the builder')
    helpers = (HERE / 'runtime_helpers.py').read_text(encoding='utf-8')
    build_info = {
        'recipe': 'v11-fixed-center-inference-guards-v1',
        'v10_sha256': hashlib.sha256(v10_path.read_bytes()).hexdigest(),
        'v9_sha256': hashlib.sha256(v9_path.read_bytes()).hexdigest(),
        'helpers_sha256': hashlib.sha256(helpers.encode()).hexdigest(),
        'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'training_enabled': False,
        'orthodiffusion_in_final_blend': False,
        'reported_baseline_auc': 0.935,
        'measured_v11_auc': None,
    }
    header = '''# RSNA Knee Ensemble V11 — pretrained inference candidate

Built from the saved V10 combined notebook, retaining its DINOv2, DINOv3,
RadImageNet and three CoAtNet arms and their blend coefficients. No new training.

- Restore exact V9 fixed-center crops for all frozen-model paths.
- Preserve valid current-run predictions after errors; never replace them with a
  benchmark fallback. Record stage outputs and hashes in `v11_diagnostics/`.
- Admit only full-window Stage-1 members with fixed, no-jitter evaluation.
- Normalize missing ensemble votes per study, and commit bank updates atomically.
- Replace the full-test RAM/future cache with a lazy, bounded temporary disk cache.
- Exclude incomplete/non-finite Stage-4 arms; block stale CoAtNet outputs on reruns.
- Save a build manifest and `v11_run_receipt.json` to make results traceable.

**Validation status:** user-reported baseline 0.935; V11 AUC unmeasured. This is not
a claim of +0.02 or a reproduced OrthoDiffusion result. OrthoDiffusion is not added
to the final blend without a validated extraction/head pipeline. Existing training
fallback is removed. Use Kaggle T4 x2 and the supplied metadata inputs.

Lineage: `mattiaangeli/bend-the-knee-to-dinov3-ensembled` plus the community
CoAtNet branch. Credit: pilkwang, mattiaangeli, marwanmath, antoinegg1, prvsiyan,
sofiaanjenje, tonylica, stevenleehans, cf696666, romantamrazov, dreaddevelopment.
Original licenses and checkpoint terms still apply; see the repository README.
'''
    set_source(cells[0], header)
    set_source(cells[1], source(cells[1]) + '''

# V11: fixed model definition and bounded cache, not AUC-tuned parameters.
V11_STAGE1_JITTER = False
V11_VOLUME_CACHE_GIB = 12
''')
    set_source(cells[8], source(cells[8]) + '''
    raise RuntimeError('V11 requires a working GPU. Select Kaggle GPU T4 x2; no CPU/training fallback.')
''')

    # Preserve exact upstream crop arithmetic, including odd-sized images/crops.
    for index, name in ((12, 'read_slot'), (28, 'read_crop'), (35, 'mm_crop_resize')):
        set_source(cells[index], replace_function(source(cells[index]), name, find_function(v9, name)))
    code = source(cells[12])
    start, end = function_span(code, '_content_crop_origin')
    lines = code.splitlines(keepends=True)
    set_source(cells[12], ''.join(lines[:start] + lines[end:]))

    code = replace_function(source(cells[18]), '_combine', '''
def _combine(per_member):
    return v11_combine_members(per_member, TARGETS)
''')
    code = replace_function(code, 'bank', '''
def bank(m, ids, pred, starts, jitter, public_pred=None, public_soft=None):
    if list(starts) != list(starts_full) or bool(jitter) != bool(V11_STAGE1_JITTER):
        raise WeightsError(f"{m['id']}: incomplete or changed window definition")
    if not np.isfinite(pred).all():
        raise WeightsError(f"{m['id']}: non-finite predictions")
    if float(np.std(pred)) < 1e-09:
        log(f"  {m['id']}: degenerate predictions; not banked")
        return
    with BANK_LOCK:
        # Retrying a failed save must replace, not double-count, this member.
        candidate = [p for p in per_member if p['id'] != m['id']]
        candidate.append({'id': m['id'], 'fold': m.get('fold'), 'ids': ids,
                          'pred': pred, 'weight': m.get('weight', 1.0),
                          'target_weight': m.get('target_weight'), 'holdout': m.get('holdout')})
        frontier = [p for p in public_frontier_members if p['id'] != m['id']]
        if public_pred is not None:
            if (not np.isfinite(public_pred).all() or public_soft is None
                    or not np.isfinite(public_soft).all() or float(np.std(public_pred)) < 1e-09):
                raise WeightsError(f"{m['id']}: invalid public-frontier prediction")
            frontier.append({'id': m['id'], 'fold': m.get('fold'), 'ids': ids,
                             'pred': public_pred, 'soft_pred': public_soft})
        all_ids, acc = _combine(candidate)
        write_submission(acc, all_ids, test_df, 'submission.csv')
        per_member[:] = candidate
        public_frontier_members[:] = frontier
        V11_RUN.member_committed(m['id'], windows=len(starts), jitter=bool(jitter),
                                 checkpoint=str(m.get('file', '')), pixel_group=str(m.get('pixel_group', '')))
        log(f"  banked {m['id']} with all {len(starts)} windows; {len(per_member)} complete member(s)")
''')
    code = replace_function(code, 'pop_next', '''
def pop_next():
    with SCHED_LOCK:
        if not pending:
            return (None, None, False)
        left = min(TIME_BUDGET - (time.time() - T0), wall_left())
        if not v11_member_fits(left, len(pending) + left_after, len(DEVS),
                               est['fixed'], est['win'], len(starts_full), V11_STAGE1_JITTER):
            V11_RUN.note('stage1_budget_skip', members=[str(m['id']) for m in pending],
                         required_windows=len(starts_full))
            pending.clear()
            return (None, None, False)
        return (pending.pop(0), list(starts_full), bool(V11_STAGE1_JITTER))
''')
    code = replace_once(code, "no member produced predictions; submission stays at 0.5",
                         "no complete member produced predictions; refusing a fake submission")
    code = replace_once(code, 'def infer_from_package(path, dev=None):\n',
                         'def infer_from_package(path, dev=None):\n    global V11_PUBLIC_FRONTIER_READY\n    V11_PUBLIC_FRONTIER_READY = False\n')
    code = replace_once(code, "        frontier_sub = write_submission(frontier_acc, frontier_ids, test_df, 'submission_public_0899.csv')",
                         "        frontier_sub = write_submission(frontier_acc, frontier_ids, test_df, 'submission_public_0899.csv')\n        V11_PUBLIC_FRONTIER_READY = True")
    set_source(cells[18], code)

    code = replace_function(source(cells[22]), 'write_submission', '''
def write_submission(pred, studies, test_df, path):
    values = np.asarray(pred)
    ids = list(map(str, studies))
    if values.shape != (len(ids), len(TARGETS)) or not np.isfinite(values).all():
        raise WeightsError('invalid member/ensemble matrix')
    if len(ids) != len(set(ids)):
        raise WeightsError('duplicate study IDs in ensemble matrix')
    sub = pd.DataFrame(pd.DataFrame(values).rank(pct=True).values, columns=TARGETS)
    sub.insert(0, 'StudyInstanceUID', ids)
    expected = test_df[['StudyInstanceUID']].astype(str)
    if set(ids) - set(expected['StudyInstanceUID']):
        raise WeightsError('unexpected study IDs in ensemble matrix')
    sub = expected.merge(sub, on='StudyInstanceUID', how='left', validate='one_to_one')
    # Preserve the inherited missing-image fallback, but never hide non-finite
    # model outputs (rejected above). Such missing rows are visible in the receipt.
    missing = int(sub[TARGETS].isna().any(axis=1).sum())
    if missing:
        V11_RUN.note('stage1_missing_image_rows', count=missing)
    sub[TARGETS] = sub[TARGETS].fillna(0.5)
    v11_validate_submission(sub, expected['StudyInstanceUID'], TARGETS)
    v11_atomic_csv(sub, path)
    return sub
''')
    code = replace_function(code, 'write_benchmark_submission', '''
def write_benchmark_submission():
    raise RuntimeError('V11 never creates benchmark submissions')
''')
    code = replace_once(code, 'def main():\n    write_benchmark_submission()\n', 'def main():\n')
    code = replace_once(code, '            public.to_csv(native_path, index=False)',
                         '            v11_atomic_csv(public, native_path)')
    code = replace_once(code, "            public_path = Path('submission_public_0899.csv')",
                         "            public_path = Path('submission_public_0899.csv')\n            if not V11_PUBLIC_FRONTIER_READY:\n                raise WeightsError('public-frontier output was not produced in this run')")
    training_start = code.index('    # This path only runs if REQUIRE_WEIGHTS')
    code = code[:training_start] + "    raise WeightsError('V11 is inference-only: attach the pretrained weights; training is disabled')\n"
    set_source(cells[22], code)
    set_source(cells[23], '''V11_RUN = V11RunAudit(ROOT, '/kaggle/working', TARGETS, V11_BUILD_INFO)
try:
    main()
except LabelSourceError:
    traceback.print_exc()
    raise
except Exception as _v11_stage1_error:
    traceback.print_exc()
    V11_RUN.recover('stage1', _v11_stage1_error)
V11_RUN.snapshot('stage1')
log('done')
''')

    # No stage may silently consume an invalid file emitted by its predecessor.
    set_source(cells[31], source(cells[31]) + "\nV11_RUN.snapshot('stage2')\n")
    set_source(cells[33], source(cells[33]) + "\nV11_RUN.snapshot('stage3')\n")

    code = source(cells[35])
    code = replace_function(code, 'rankpct', '''
def rankpct(x):
    return v11_rankpct(x)
''')
    cache_start = code.index('        # All arms share IMG/SLOTS/CROP_MM')
    cache_end = code.index('        # SEQUENTIAL ARMS', cache_start)
    code = code[:cache_start] + '''        # Lazy shared cache: one decoded study live, reusable arrays on bounded disk.
        _vol_cache = V11VolumeCache(max_bytes=int(V11_VOLUME_CACHE_GIB * 1024**3))

        def _study_volume(sid):
            return _vol_cache.get(sid, lambda uid: build_study(uid, SER, tsdir, reader))

''' + code[cache_end:]
    code = replace_once(code, '                        arm_probs[a][i] = infer_probs(model, xw, dev)', '''                        _arm_prediction = np.asarray(infer_probs(model, xw, dev))
                        if _arm_prediction.shape != (len(LAB),) or not np.isfinite(_arm_prediction).all():
                            raise ValueError('invalid CoAtNet study prediction')
                        arm_probs[a][i] = _arm_prediction''')
    code = replace_once(code,
        '                        print(f"  [arm {a}] study {i} {sid[:16]} FALLBACK ({type(e).__name__}: {e})", flush=True)',
        '''                        print(f"  [arm {a}] study {i} failed; incomplete arm excluded ({type(e).__name__}: {e})", flush=True)
                        V11_RUN.note('stage4_arm_failed', arm=arm['file'], row=i,
                                     error=f'{type(e).__name__}: {e}')
                        arm_finished = False
                        break''')
    code = replace_once(code, '                    completed.append(a)', '''                    completed.append(a)
                    V11_RUN.note('stage4_arm_completed', arm=arm['file'], studies=N, resolution=int(res))''')
    start = code.index('        completed = []')
    end = code.index('        del _vol_cache\n        gc.collect()\n', start)
    arm_block = code[start:end]
    code = code[:start] + '        try:\n' + textwrap.indent(arm_block, '    ') + '''        finally:
            try:
                V11_RUN.note('stage4_volume_cache', **_vol_cache.stats())
            finally:
                _vol_cache.close()
        gc.collect()
''' + code[end + len('        del _vol_cache\n        gc.collect()\n'):]
    code = replace_once(code, '            ranks[~np.isfinite(ranks)] = 0.5',
                         "            raise ValueError('non-finite CoAtNet blend; preserving previous stage')")
    code = 'V11_COATNET_READY = False\n' + code
    code = replace_once(code, '            main()\n        except Exception as _coat_exc:',
                         '            main()\n            V11_COATNET_READY = True\n        except Exception as _coat_exc:')
    code = replace_once(code, '    if _blend_coatnet_path.is_file():',
                         '    if V11_COATNET_READY and _blend_coatnet_path.is_file():')
    code += "\nV11_RUN.snapshot('stage4')\n"
    set_source(cells[35], code)
    set_source(cells[37], source(cells[37]) + "\nV11_RUN.snapshot('final')\nprint('V11 receipts: /kaggle/working/v11_run_receipt.json')\n")

    # Inline the helper module; uploading the .ipynb alone needs no local imports.
    helper_cell = code_cell('V11_BUILD_INFO = ' + repr(build_info) + '\n\n' + helpers,
                            'v11-runtime-guards')
    notebook['cells'].insert(2, helper_cell)
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            compile(source(cell), f'V11-cell-{index}', 'exec')
            cell['outputs'] = []
            cell['execution_count'] = None
    notebook['metadata']['v11_build'] = build_info
    encoded = json.dumps(notebook, indent=1, ensure_ascii=False) + '\n'
    output = HERE / 'rsna-knee-ensemble-v11.ipynb'
    output.write_text(encoded, encoding='utf-8')

    metadata = json.loads((ROOT / 'v10' / 'kernel-metadata.json').read_text())
    owner = metadata['id'].split('/')[0]
    metadata.update(id=owner + '/rsna-knee-ensemble-v11', title='RSNA Knee Ensemble V11',
                    code_file=output.name)
    (HERE / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (HERE / 'build_manifest.json').write_text(json.dumps({**build_info,
        'notebook_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'code_cells_compiled': sum(c['cell_type'] == 'code' for c in notebook['cells'])}, indent=2) + '\n')
    return notebook


if __name__ == '__main__':
    result = build()
    print(f'Built {HERE / "rsna-knee-ensemble-v11.ipynb"}: {len(result["cells"])} cells; compilation passed')
