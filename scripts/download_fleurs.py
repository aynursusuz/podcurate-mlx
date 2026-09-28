"""Fetch four short public test clips per language from a pinned FLEURS revision.

Only the beginning of each original tar stream is read. The resulting read-speech
fixtures exercise software, not podcast or synthetic-generator quality.
"""

import argparse
import concurrent.futures
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
BASE = f"https://huggingface.co/datasets/google/fleurs/resolve/{REVISION}"
CONFIGS = [("en", "en_us"), ("zh", "cmn_hans_cn"), ("ja", "ja_jp"), ("tr", "tr_tr")]


def fetch(item, output):
    language, config = item
    with urllib.request.urlopen(f"{BASE}/data/{config}/test.tsv", timeout=60) as response:
        rows = [line.split("\t") for line in response.read().decode().splitlines()]
    lookup = {row[1]: row for row in rows}
    url = f"{BASE}/data/{config}/audio/test.tar.gz"
    records = []
    with (
        urllib.request.urlopen(url, timeout=90) as response,
        tarfile.open(fileobj=response, mode="r|gz") as archive,
    ):
        for member in archive:
            name = Path(member.name).name
            if not member.isfile() or name not in lookup:
                continue
            row = lookup[name]
            frames = int(row[5])
            if not 3 * 16000 <= frames <= 27 * 16000:
                continue
            # Read an individual bounded member; never extract paths from the archive.
            if member.size > 10_000_000:
                raise ValueError("unexpected fixture size")
            data = archive.extractfile(member).read()
            path = output / f"{language}-{len(records)}.wav"
            path.write_bytes(data)
            records.append(
                {
                    "id": f"fleurs-{language}-{row[0]}-{len(records)}",
                    "audio": path.name,
                    "language": language,
                    "kind": "podcast",
                    "reference_text": row[2],
                    "source_id": f"fleurs:{config}:test:{name}",
                    "dataset": "google/fleurs",
                    "dataset_revision": REVISION,
                    "license": "CC-BY-4.0",
                    "source_url": url,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "original_name": name,
                }
            )
            if len(records) == 4:
                break
    if len(records) != 4:
        raise ValueError(f"missing fixtures: {language}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        groups = list(pool.map(lambda item: fetch(item, args.out), CONFIGS))
    with (args.out / "manifest.jsonl").open("w", encoding="utf8") as f:
        for records in groups:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"records": 16, "revision": REVISION}))


if __name__ == "__main__":
    main()
