"""Local download handoff, CRC-checked extraction, preparation and first fold.

Run with the project's .venv-v16 interpreter. This launcher does not alter V16's
model or preprocessing contract. Data and logs stay outside version control.
"""
import argparse
import ctypes
import json
import msvcrt
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
LOCAL = PROJECT / 'local'
DATA = Path('D:/RSNA-Knee/data')
WORK = LOCAL / 'work-laptop'


def status(stage, **details):
    record = dict(stage=stage, time=time.strftime('%Y-%m-%d %H:%M:%S'), **details)
    path = LOCAL / 'status.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(record, indent=2), encoding='utf-8')
    os.replace(temp, path)
    print(json.dumps(record), flush=True)


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=PROJECT, check=True)


def wait_download(pid):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x00100000, False, pid)
    if handle:
        try:
            while kernel.WaitForSingleObject(handle, 30000) == 258:
                pass
        finally:
            kernel.CloseHandle(handle)


def extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        members = [m for m in source.infolist() if not m.is_dir()]
        paths = []
        for member in members:
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError('Archive contains an unsafe path')
            paths.append(target)
        required = sum(m.file_size for m, p in zip(members, paths) if not p.exists())
        if shutil.disk_usage(root).free < required + 20 * 2**30:
            raise RuntimeError('Insufficient extraction space with 20 GiB reserve')
        for index, (member, target) in enumerate(zip(members, paths)):
            # Files created here are atomically published only after ZIP CRC validation.
            if target.exists():
                if target.stat().st_size != member.file_size:
                    raise RuntimeError(f'Existing file size mismatch: {target}')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + '.partial')
            with source.open(member) as src, partial.open('wb') as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            os.replace(partial, target)
            if index % 5000 == 0:
                status('extracting', completed=index + 1, files=len(members))


def main():
    LOCAL.mkdir(parents=True, exist_ok=True)
    # Held for this process lifetime; prevents a VS Code task duplicating the job.
    lock = (LOCAL / 'pipeline.lock').open('a+b')
    if os.fstat(lock.fileno()).st_size == 0:
        lock.write(b'0')
        lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        print('The laptop pipeline is already running.', flush=True)
        return
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--skip-download', action='store_true')
    args = parser.parse_args()
    os.environ['USE_TF'] = '0'
    os.environ['PYTHONUNBUFFERED'] = '1'
    archive = LOCAL / 'downloads/rsna-knee-abnormality-detection.zip'
    if args.wait_pid:
        status('downloading', download_pid=args.wait_pid)
        wait_download(args.wait_pid)
    elif not args.skip_download:
        status('downloading')
        run('-m', 'kaggle', 'competitions', 'download',
            'rsna-knee-abnormality-detection', '-p', LOCAL / 'downloads')
    # An incomplete/failed download cannot pass ZIP central-directory validation.
    status('checking_archive')
    if not zipfile.is_zipfile(archive):
        raise RuntimeError('Download incomplete; inspect download.err.log before restarting')
    status('extracting')
    extract(archive, DATA)
    status('preparing')
    run('-m', 'v16', 'prepare', '--data-root', DATA, '--work', WORK,
        '--config', LOCAL / 'config.laptop.json',
        '--report-labels', LOCAL / 'labels/llm_labels_v2.csv')
    receipt = json.loads((WORK / 'preparation.json').read_text())
    if receipt['missing_images']:
        raise RuntimeError('Some training studies have no usable images; inspect preparation.json')
    status('training', fold=0, arm='platt', grouping='study-only; patient separation unverified')
    run('-m', 'v16', 'train', '--work', WORK, '--fold', '0', '--seed', '1400', '--arm', 'platt')
    status('first_fold_complete', output=str(WORK / 'runs/seed1400/platt/fold0'),
           note='One fold is a pilot, not a validated leaderboard improvement.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error=str(exc))
        raise
