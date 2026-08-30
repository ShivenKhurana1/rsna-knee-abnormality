#!/usr/bin/env bash
# Downloads everything the local training run needs. ~22 GB, mostly the corpus.
# Requires: kaggle CLI authenticated, and the competition rules accepted on that account.
set -euo pipefail
DATA="${1:-./data}"
mkdir -p "$DATA"; cd "$DATA"

echo "==> preprocessed corpus, part 1 of 2 (~16 GB)"
kaggle datasets download dreaddevelopment/knee-raptor-corpus --unzip -p .

echo "==> preprocessed corpus, part 2 of 2 (~6 GB)"
kaggle datasets download dreaddevelopment/knee-raptor-corpus-ext --unzip -p .

echo "==> LLM soft labels (4,349 studies)"
kaggle datasets download dreaddevelopment/rsna-knee-labels --unzip -p .

echo "==> competition train.csv (for the 58 expert labels used as the validation gate)"
kaggle competitions download rsna-knee-abnormality-detection -f train.csv -p .
[ -f train.csv.zip ] && unzip -o train.csv.zip && rm -f train.csv.zip

echo "==> merging the two corpus parts into one array"
python3 - <<'PY'
import numpy as np, os
if os.path.exists('all_vols_merged.npy'):
    print('  already merged'); raise SystemExit
a=np.load('all_vols.npy',mmap_mode='r'); b=np.load('extra_vols.npy',mmap_mode='r')
out=np.lib.format.open_memmap('all_vols_merged.npy',mode='w+',dtype=a.dtype,
                              shape=(a.shape[0]+b.shape[0],)+a.shape[1:])
CH=200
for s in range(0,a.shape[0],CH): out[s:s+CH]=a[s:s+CH]
for s in range(0,b.shape[0],CH): out[a.shape[0]+s:a.shape[0]+s+CH]=b[s:s+CH]
out.flush(); del out
np.save('all_masks_merged.npy', np.concatenate([np.load('all_masks.npy'),np.load('extra_masks.npy')],0))
np.save('all_ids_merged.npy',   np.concatenate([np.load('all_ids.npy',allow_pickle=True).astype(str),
                                                np.load('extra_ids.npy',allow_pickle=True).astype(str)]))
for s,d in (('all_vols_merged.npy','all_vols.npy'),('all_masks_merged.npy','all_masks.npy'),
            ('all_ids_merged.npy','all_ids.npy')):
    os.replace(s,d)
print('  merged ->', np.load('all_vols.npy',mmap_mode='r').shape)
PY
echo "==> done. corpus in $(pwd)"
