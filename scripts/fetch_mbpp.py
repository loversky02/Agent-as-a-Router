#!/usr/bin/env python3
"""Download the sanitized MBPP dataset to data/mbpp.json.

macOS-friendly: tries `curl` first (it ships with a working CA store), then falls
back to urllib with certifi, then an unverified context as a last resort — MBPP is a
public file, so the fallback is a dev convenience, not a security decision.
"""
import json
import os
import ssl
import subprocess
import urllib.request

URL = ("https://raw.githubusercontent.com/google-research/google-research/"
       "master/mbpp/sanitized-mbpp.json")
OUT = "data/mbpp.json"


def _via_curl():
    subprocess.run(["curl", "-fsSL", URL, "-o", OUT], check=True)


def _via_urllib():
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        print("  (warning: falling back to unverified TLS to fetch a public file)")
        ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(URL, timeout=30, context=ctx) as r:
        with open(OUT, "wb") as f:
            f.write(r.read())


def main():
    os.makedirs("data", exist_ok=True)
    print(f"Downloading MBPP -> {OUT} ...")
    try:
        _via_curl()
    except Exception as e:
        print(f"  (curl unavailable: {e}; trying urllib)")
        _via_urllib()
    with open(OUT) as f:
        data = json.load(f)
    print(f"Saved {len(data)} problems -> {OUT}")


if __name__ == "__main__":
    main()
