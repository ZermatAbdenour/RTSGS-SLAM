#!/usr/bin/env python3

import os
import re
import argparse
import requests
from concurrent.futures import ThreadPoolExecutor

# =========================================================
# 3RScan URLs
# =========================================================
BASE_URL = 'http://campar.in.tum.de/public_datasets/3RScan/'
DATA_URL = BASE_URL + 'Dataset/'

RELEASE = 'release_scans.txt'
HIDDEN_RELEASE = 'test_rescans.txt'

# =========================================================
# FILE TYPES
# =========================================================
TEST_FILETYPES = [
    'mesh.refined.v2.obj',
    'mesh.refined.mtl',
    'mesh.refined_0.png',
    'sequence.zip'
]

FILETYPES = TEST_FILETYPES + [
    'labels.instances.annotated.v2.ply',
    'mesh.refined.0.010000.segs.v2.json',
    'semseg.v2.json'
]

id_reg = re.compile(
    r"[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}"
)

# =========================================================
# LOG FILE
# =========================================================
LOG_FILE = "completed_scans.txt"


def load_completed():
    if not os.path.exists(LOG_FILE):
        return set()
    with open(LOG_FILE, "r") as f:
        return set(line.strip() for line in f.readlines())


def mark_completed(scan_id):
    with open(LOG_FILE, "a") as f:
        f.write(scan_id + "\n")


# =========================================================
# SCAN LIST
# =========================================================
def get_scans(url):
    import urllib.request as urllib
    scans = []
    for line in urllib.urlopen(url):
        line = line.decode().strip()
        m = id_reg.search(line)
        if m:
            scans.append(m.group())
    return scans


# =========================================================
# DOWNLOAD FILE
# =========================================================
def download_file(url, out_file, retries=3):
    if os.path.exists(out_file):
        return True

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    for attempt in range(retries):
        try:
            r = requests.get(url, stream=True, timeout=30)
            r.raise_for_status()

            tmp_file = out_file + ".tmp"

            with open(tmp_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

            os.rename(tmp_file, out_file)
            return True

        except Exception as e:
            print(f"[Retry {attempt+1}] {url} failed: {e}")

    print(f"[FAILED] {url}")
    return False


# =========================================================
# DOWNLOAD SINGLE SCAN
# =========================================================
def download_scan(scan_id, out_dir, file_types, workers=8):
    scan_dir = os.path.join(out_dir, scan_id)
    os.makedirs(scan_dir, exist_ok=True)

    def task(ft):
        url = f"{DATA_URL}/{scan_id}/{ft}"
        out_file = os.path.join(scan_dir, ft)
        return download_file(url, out_file)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(task, file_types))

    return all(results)


# =========================================================
# DOWNLOAD DATASET
# =========================================================
def download_release(scans, out_dir, file_types, scan_workers=4, file_workers=8):
    print(f"Downloading {len(scans)} scans...")

    completed = load_completed()

    def task(scan_id):
        if scan_id in completed:
            print(f"[SKIP] already completed {scan_id}")
            return

        ok = download_scan(scan_id, out_dir, file_types, workers=file_workers)

        if ok:
            mark_completed(scan_id)
            print(f"[DONE] {scan_id}")
        else:
            print(f"[INCOMPLETE] {scan_id}")

    with ThreadPoolExecutor(max_workers=scan_workers) as ex:
        list(ex.map(task, scans))


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--out_dir", required=True)
    parser.add_argument("--id")
    parser.add_argument("--type", nargs="+")
    parser.add_argument("--scan_workers", type=int, default=4)
    parser.add_argument("--file_workers", type=int, default=8)

    # 🔥 NEW LIMIT ARGUMENT
    parser.add_argument("--max_scans", type=int, default=None,
                        help="limit number of scans to download")

    args = parser.parse_args()

    print("🚀 3RScan downloader (fast + limit + resume support)")
    input("Press ENTER to continue...")

    # -----------------------------
    # LOAD SCANS
    # -----------------------------
    release_scans = get_scans(BASE_URL + RELEASE)
    test_scans = get_scans(BASE_URL + HIDDEN_RELEASE)

    # -----------------------------
    # APPLY LIMIT (NEW)
    # -----------------------------
    if args.max_scans is not None:
        release_scans = release_scans[:args.max_scans]
        test_scans = test_scans[:args.max_scans]

        print(f"[LIMIT] Using only {args.max_scans} scans per split")

    # -----------------------------
    # FILE TYPES
    # -----------------------------
    file_types = FILETYPES
    file_types_test = TEST_FILETYPES

    if args.type:
        file_types = args.type
        file_types_test = [t for t in args.type if t in TEST_FILETYPES]

    # -----------------------------
    # SINGLE SCAN MODE
    # -----------------------------
    if args.id:
        download_scan(args.id, args.out_dir, file_types, workers=args.file_workers)
        return

    # -----------------------------
    # FULL DOWNLOAD MODE
    # -----------------------------
    print("Starting download...")
    print(f"Scan workers: {args.scan_workers}, File workers: {args.file_workers}")

    download_release(
        release_scans,
        args.out_dir,
        file_types,
        scan_workers=args.scan_workers,
        file_workers=args.file_workers
    )

    download_release(
        test_scans,
        args.out_dir,
        file_types_test,
        scan_workers=args.scan_workers,
        file_workers=args.file_workers
    )


if __name__ == "__main__":
    main()