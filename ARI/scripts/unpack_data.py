"""Unpack the data files that are stored compressed and check their SHA-256.

    python ARI/scripts/unpack_data.py

A file that is already present with the right SHA-256 is left alone.
"""
import gzip
import hashlib
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILES = {  # unpacked file: (compressed file, sha256 of the unpacked file)
    "data/covertype_expanded.csv": ("data/covertype_expanded.csv.gz",
                                    "ab286fca6639da51e7f2659e3faa0015096543ab0ff97b5db16948b08121ae26"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    failed = False
    for rel, (packed, want) in FILES.items():
        path, src = os.path.join(ROOT, rel), os.path.join(ROOT, packed)
        if os.path.exists(path) and sha256(path) == want:
            print(f"[OK  ] {rel}")
            continue
        if not os.path.exists(src):
            print(f"[FAIL] {rel}: {packed} not found")
            failed = True
            continue
        tmp = path + ".part"
        with gzip.open(src, "rb") as fin, open(tmp, "wb") as fout:
            shutil.copyfileobj(fin, fout, 1 << 20)
        if sha256(tmp) != want:
            os.remove(tmp)
            print(f"[FAIL] {rel}: SHA-256 of the unpacked file differs from the expected one")
            failed = True
            continue
        os.replace(tmp, path)
        print(f"[OK  ] {rel} (unpacked from {packed})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
