"""Package a private, first-fold Kaggle GPU pilot with attached official inputs."""
import json
from pathlib import Path
from v16.build_notebooks import code, markdown, MODULES

HERE = Path(__file__).resolve().parent
DEST = HERE / 'kaggle_gpu'
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
body = '''import shutil, pandas as pd
from v16.common import config
from v16.runner import prepare, train, finite_optimizer_step
from v16.model import KneeModel, masked_loss
from v16.data import collate, to_device
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
cfg=config(runtime/'v16/config.json')
cfg.update(encoder_path=str(models[0]),image_size=224,max_centres=8,max_series=3,
           encoder_chunk=4,device='cuda')
# Check the actual allocated GPU BEFORE spending hours on image preparation.
targets=list(pd.read_csv(roots[0]/'sample_submission.csv',nrows=0).drop(columns=cfg['id_col']).columns)
probe=KneeModel(cfg,len(targets)).cuda().train()
optimizer=torch.optim.AdamW([p for p in probe.parameters() if p.requires_grad],lr=1e-5)
scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
for step in range(20):
    batch=collate([{'uid':'numerical-probe','series':[
        {'x':torch.rand(cfg['max_centres'],3,cfg['image_size'],cfg['image_size']),
         'position':torch.linspace(0,1,cfg['max_centres']),'meta':torch.zeros(6)}
        for _ in range(cfg['max_series'])]}])
    batch=to_device(batch,torch.device('cuda'))
    for attempt in range(12):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            logits,_=probe(batch)
        truth=torch.full_like(logits,float(step%2))
        loss=masked_loss(logits,truth,torch.ones_like(logits,dtype=torch.bool),cfg)
        assert torch.isfinite(loss), 'GPU preflight loss is nonfinite'
        scaler.scale(loss).backward()
        if finite_optimizer_step(probe,optimizer,scaler,cfg['grad_clip']): break
    else: raise RuntimeError('GPU preflight could not recover from overflow')
    assert all(torch.isfinite(p).all() for p in probe.parameters())
print('GPU stability preflight passed: 20 optimizer steps',flush=True)
del probe,optimizer,scaler,batch,logits,loss
torch.cuda.empty_cache()
work=Path('/tmp/v16_work'); work.mkdir(exist_ok=True)
output=Path('/kaggle/working/v16_output'); output.mkdir(exist_ok=True)
runs=output/'runs'; runs.mkdir(exist_ok=True)
if not (work/'runs').exists():
    (work/'runs').symlink_to(runs,target_is_directory=True)
n=len(pd.read_csv(roots[0]/'train.csv'))
bound=n*cfg['max_series']*cfg['max_centres']*len(cfg['offsets'])*cfg['image_size']**2*2
free=shutil.disk_usage(work).free
assert free>bound*1.1+5*2**30, f'Cache needs up to {bound/2**30:.1f} GiB plus headroom; free {free/2**30:.1f}'
print('Preparing all training studies; scratch cache is excluded from saved output',flush=True)
receipt=prepare(roots[0],work,cfg,reports[0])
for path in work.iterdir():
    if path.is_file(): shutil.copy2(path,output/path.name)
shutil.copy2(work/'cache/manifest.json',output/'cache_manifest.json')
assert not receipt['missing_images'], 'Review missing training image diagnostics before training'
print('Training fold 0; study-only grouping, patient separation unverified',flush=True)
result=train(work,0,seed=1400,arm='platt')
print(json.dumps(result,indent=2),flush=True)
'''
nb={'nbformat':4,'nbformat_minor':5,
    'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
    'cells':[markdown('# RSNA Knee V16 GPU Pilot\nPrivate first-fold training. No score claim.\n224 px, 8 centers, up to 3 series; eight epochs. Temporary image cache; saved epoch checkpoints.\nStability fix: FP32 sequence head, bounded same-batch AMP retries, GPU preflight and batch progress logs.'),code(setup),code(embed),code(body)]}
for i,cell in enumerate(nb['cells']):
    cell['id']=f'gpu-pilot-{i}'
    if cell['cell_type']=='code': compile(''.join(cell['source']),cell['id'],'exec')
(DEST/'train.ipynb').write_text(json.dumps(nb,indent=1),encoding='utf-8')
metadata={'id':'seanzhang2445/rsna-knee-v16-gpu-pilot','title':'RSNA Knee V16 GPU Pilot',
          'code_file':'train.ipynb','language':'python','kernel_type':'notebook',
          'is_private':True,'enable_gpu':True,'enable_internet':True,
          'competition_sources':['rsna-knee-abnormality-detection'],
          'dataset_sources':['stevenleehans/rsna-knee-llm-report-labels'],
          'model_sources':['metaresearch/dinov2/PyTorch/small/1'],'kernel_sources':[]}
(DEST/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(DEST)
