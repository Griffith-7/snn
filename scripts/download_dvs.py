"""Parallel range-request downloader for the CIFAR10-DVS frame files.

Source: NDA_SNN (github.com/Intelligent-Computing-Lab-Yale/NDA_SNN) Google Drive
mirror of the preprocessed CIFAR10-DVS data (per-sample .pt tensors of frames).

Downloading is split into N_CHUNKS parallel range requests (the host supports
Content-Range: bytes 0-0/TOTAL), each written to a `.part` file; re-running the
script resumes any incomplete chunks. Final files are concatenated in order.
"""
import concurrent.futures as cf
import os
import sys

import requests

URL = "https://drive.usercontent.google.com/download"
FILES = {
    "train": ("1pzYnhoUvtcQtxk_Qmy4d2VrhWhy5R-t9", "train_file"),
    "test": ("1q1k6JJgVH3ZkHWMg2zPtrZak9jRP6ggG", "test_file"),
}
N_CHUNKS = 8
CHUNK_BYTES = 8 * 1024 * 1024


def _params(drive_id):
    return {"id": drive_id, "export": "download", "confirm": "t"}


def total_size(drive_id):
    r = requests.get(URL, params=_params(drive_id), headers={"Range": "bytes=0-0"},
                     stream=True, timeout=60)
    cr = r.headers.get("Content-Range", "")
    return int(cr.split("/")[-1])


def download_chunk(drive_id, start, end, part_path):
    if os.path.exists(part_path) and os.path.getsize(part_path) == end - start:
        return start, end - start, "cached"
    headers = {"Range": f"bytes={start}-{end - 1}"}
    r = requests.get(URL, params=_params(drive_id), headers=headers, stream=True,
                     timeout=120)
    n = 0
    with open(part_path, "wb") as f:
        for block in r.iter_content(1 << 16):
            f.write(block)
            n += len(block)
    return start, n, "downloaded"


def fetch(name):
    drive_id, out_name = FILES[name]
    root = os.path.dirname(os.path.abspath(__file__))
    dest = os.path.join(os.path.dirname(root), "data", "cifar10dvs", out_name)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    total = total_size(drive_id)
    chunks = list(range(0, total, CHUNK_BYTES)) + [total]
    bounds = [(chunks[i], chunks[i + 1]) for i in range(len(chunks) - 1)]
    print(f"[{name}] total {total} bytes in {len(bounds)} chunks -> {dest}")
    jobs = []
    with cf.ThreadPoolExecutor(max_workers=N_CHUNKS) as ex:
        for i, (a, b) in enumerate(bounds):
            part = f"{dest}.part{i}"
            jobs.append(ex.submit(download_chunk, drive_id, a, b, part))
        for j in cf.as_completed(jobs):
            start, n, how = j.result()
            print(f"[{name}] chunk @{start} {n} B ({how})")
    with open(dest, "wb") as out:
        for i in range(len(bounds)):
            with open(f"{dest}.part{i}", "rb") as p:
                while True:
                    blk = p.read(1 << 20)
                    if not blk:
                        break
                    out.write(blk)
    got = os.path.getsize(dest)
    print(f"[{name}] assembled {got} bytes (expected {total}) "
          f"{'OK' if got == total else 'MISMATCH'}")
    if got == total:
        for i in range(len(bounds)):
            os.remove(f"{dest}.part{i}")
    return got, total


if __name__ == "__main__":
    names = sys.argv[1:] or list(FILES)
    for nm in names:
        got, total = fetch(nm)
        if got != total:
            sys.exit(f"[{nm}] failed: {got}/{total}")
