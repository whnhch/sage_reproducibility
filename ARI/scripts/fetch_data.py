"""Download the inputs that are too large for GitHub from Zenodo and check their SHA-256.

    python ARI/scripts/fetch_data.py

A file that is already present with the right SHA-256 is left alone.
"""
import hashlib
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ZENODO_RECORD = ""  # TODO (authors): the Zenodo record ID, e.g. "1234567"
FILES = {  # path in the repository: sha256
    "data/covertype_expanded.csv": "ab286fca6639da51e7f2659e3faa0015096543ab0ff97b5db16948b08121ae26",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest):
    tmp = dest + ".part"
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total, done = int(r.headers.get("Content-Length", 0)), 0
        for chunk in iter(lambda: r.read(1 << 20), b""):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 2**20:.0f} / {total / 2**20:.0f} MB", end="", flush=True)
    print()
    return tmp


def main():
    failed = False
    for rel, want in FILES.items():
        path = os.path.join(ROOT, rel)
        if os.path.exists(path) and sha256(path) == want:
            print(f"[OK  ] {rel}")
            continue
        if not ZENODO_RECORD:
            print(f"[FAIL] {rel}: missing, and no Zenodo record is set in {os.path.relpath(__file__, ROOT)}")
            failed = True
            continue
        url = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{os.path.basename(rel)}?download=1"
        print(f"[GET ] {rel} <- {url}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = download(url, path)
        if sha256(tmp) != want:
            os.remove(tmp)
            print(f"[FAIL] {rel}: SHA-256 differs from the expected one")
            failed = True
            continue
        os.replace(tmp, path)
        print(f"[OK  ] {rel}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
