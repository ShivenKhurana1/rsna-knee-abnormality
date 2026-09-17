"""Package a private, small-cohort Kaggle GPU calibration run at the FULL default
recipe (336px, up to 8 series, up to 24 centers per series -- no shrinking).

Unlike build_kaggle_gpu.py (which trains fold 0 of the entire ~4,349-study pool
at a deliberately shrunk 224/3/8 config), this packages a 208-study subset
(150 random pool studies + all 58 expert-labelled studies, matching this
project's existing dev-cohort convention) so the FULL-resolution recipe's real
per-study decode/train cost can be measured cheaply before committing a whole
Kaggle session to the full ~4,349-study pool at full resolution.
"""
import json
from pathlib import Path
from v16.build_notebooks import code, markdown, MODULES

HERE = Path(__file__).resolve().parent
DEST = HERE / 'kaggle_gpu_calib'
DEST.mkdir(exist_ok=True)
embed = "import os, sys, json\nfrom pathlib import Path\n"
embed += "os.environ['USE_TF']='0'\nos.environ['HF_HUB_OFFLINE']='1'\n"
embed += "runtime=Path('/kaggle/working/runtime'); (runtime/'v16').mkdir(parents=True,exist_ok=True)\n"
for name in MODULES + ('config.json',):
    embed += f"(runtime/'v16'/{name!r}).write_text({(HERE/name).read_text(encoding='utf-8')!r}, encoding='utf-8')\n"
embed += "sys.path.insert(0,str(runtime))\n"
setup = '''import subprocess, sys, os
os.environ['USE_TF']='0'
subprocess.check_call([sys.executable,'-m','pip','install','--quiet',
    'pydicom==3.0.2','pylibjpeg>=2,<3','pylibjpeg-libjpeg>=2,<3',
    'pylibjpeg-openjpeg>=2,<3','transformers==4.57.6'])
import torch
assert torch.cuda.is_available(), 'GPU allocation required; refusing CPU training'
torch.set_num_threads(2)
print('GPU:',torch.cuda.get_device_name(0),flush=True)
'''
body = '''import time, shutil, pandas as pd, numpy as np
from v16.common import config
from v16.runner import prepare, train
roots=[]; models=[]; reports=[]
for folder, dirs, names in os.walk('/kaggle/input'):
    if 'train.csv' in names and 'sample_submission.csv' in names:
        roots.append(Path(folder))
    if 'config.json' in names and ('pytorch_model.bin' in names or 'model.safetensors' in names):
        models.append(Path(folder))
    if 'llm_labels_v2.csv' in names:
        reports.append(Path(folder)/'llm_labels_v2.csv')
    dirs[:]=[d for d in dirs if d not in ('train_series','test_series')]
assert len(roots)==len(models)==len(reports)==1, (roots,models,reports)
real_root=roots[0]
cfg=config(runtime/'v16/config.json')
cfg.update(encoder_path=str(models[0]),device='cuda')
print('Full recipe in force: image_size=%d max_series=%d max_centres=%d encoder_chunk=%d'
      %(cfg['image_size'],cfg['max_series'],cfg['max_centres'],cfg['encoder_chunk']),flush=True)

# --- Build a 208-study scratch root (150 random pool + all 58 expert studies),
# reusing this project's existing dev-cohort convention. No image bytes are
# copied: train_series/ is symlinked whole, so build_cache only ever touches
# the subset actually listed in the scratch train.csv/train_series.csv.
train=pd.read_csv(real_root/'train.csv',dtype={'StudyInstanceUID':str})
targets=[c for c in train.columns if c not in ('StudyInstanceUID','Report')]
gold=train[train[targets].notna().any(axis=1)]
pool=train[~train['StudyInstanceUID'].isin(gold['StudyInstanceUID'])]
rng=np.random.default_rng(1400)
pool_sample=pool.iloc[rng.permutation(len(pool))[:150]]
subset=pd.concat([gold,pool_sample]).reset_index(drop=True)
print(f'Calibration cohort: {len(gold)} expert + {len(pool_sample)} pool = {len(subset)} studies',flush=True)

scratch=Path('/kaggle/working/v16_calib_root'); scratch.mkdir(exist_ok=True)
shutil.copy2(real_root/'sample_submission.csv',scratch/'sample_submission.csv')
subset.to_csv(scratch/'train.csv',index=False)
series=pd.read_csv(real_root/'train_series.csv',dtype={'StudyInstanceUID':str})
series[series['StudyInstanceUID'].isin(subset['StudyInstanceUID'])].to_csv(scratch/'train_series.csv',index=False)
link=scratch/'train_series'
if not link.exists():
    link.symlink_to(real_root/'train_series',target_is_directory=True)
# aligned() requires the report table's ids to exactly match the prepared cohort,
# so the full-pool report-labels CSV must be filtered to this subset too.
full_reports=pd.read_csv(reports[0],dtype={'StudyInstanceUID':str})
report_path=scratch/'report_labels_subset.csv'
full_reports[full_reports['StudyInstanceUID'].isin(subset['StudyInstanceUID'])].to_csv(report_path,index=False)

work=Path('/kaggle/working/v16_calib_work'); work.mkdir(exist_ok=True)
t0=time.time()
receipt=prepare(scratch,work,cfg,report_path)
prep_seconds=time.time()-t0
assert not receipt['missing_images'], 'Review missing training image diagnostics before training'
per_study=prep_seconds/receipt['studies']
print(f'Calibration prepare: {prep_seconds:.0f}s for {receipt["studies"]} studies '
      f'({per_study:.2f}s/study). Full ~4,407-study pool at this recipe extrapolates to '
      f'~{per_study*4407/3600:.1f}h of prepare alone.',flush=True)

t0=time.time()
result=train(work,0,seed=1400,arm='platt')
train_seconds=time.time()-t0
print(f'Calibration training (8 epochs, {len(subset)}-study cohort): {train_seconds:.0f}s '
      f'({train_seconds/8:.0f}s/epoch on this {len(subset)}-study cohort). A full ~4,407-study '
      f'pool epoch at this recipe extrapolates to roughly '
      f'~{train_seconds/8*4407/len(subset)/3600:.1f}h/epoch ({train_seconds/8*4407/len(subset)*8/3600:.1f}h for 8 epochs), '
      f'though per-study cost is not perfectly linear (fixed per-batch overhead, warmup).',flush=True)
print(json.dumps(result,indent=2),flush=True)
print('CAVEAT: this cohort has only 150 pool studies backing the platt/report-auxiliary '
      'policy (vs ~4,200 in a real run) -- auxiliary supervision here is far weaker than '
      'a full run. Treat metrics as a throughput/series-coverage calibration signal only, '
      'not a promotion decision.',flush=True)
'''
nb={'nbformat':4,'nbformat_minor':5,
    'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
    'cells':[markdown('# RSNA Knee V16 Full-Recipe Calibration\n'
                       '208-study cohort (150 random pool + all 58 expert studies) at the '
                       'FULL default recipe (336px, up to 8 series, up to 24 centers) -- no '
                       'shrinking. Measures real per-study prepare and per-epoch train cost at '
                       'full resolution before committing a whole session to the ~4,349-study '
                       'pool. No score claim; not a promotion decision.'),
             code(setup),code(embed),code(body)]}
for i,cell in enumerate(nb['cells']):
    cell['id']=f'calib-{i}'
    if cell['cell_type']=='code': compile(''.join(cell['source']),cell['id'],'exec')
(DEST/'train.ipynb').write_text(json.dumps(nb,indent=1),encoding='utf-8')
metadata={'id':'seanzhang2445/rsna-knee-v16-fullrecipe-calibration','title':'RSNA Knee V16 Full-Recipe Calibration',
          'code_file':'train.ipynb','language':'python','kernel_type':'notebook',
          'is_private':True,'enable_gpu':True,'enable_internet':True,
          'competition_sources':['rsna-knee-abnormality-detection'],
          'dataset_sources':['stevenleehans/rsna-knee-llm-report-labels'],
          'model_sources':['metaresearch/dinov2/PyTorch/small/1'],'kernel_sources':[]}
(DEST/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(DEST)
