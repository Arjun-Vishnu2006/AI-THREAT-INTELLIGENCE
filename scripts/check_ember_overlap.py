import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data_dir = root / "data" / "file_dataset" / "ember2018"

train_files = sorted(data_dir.glob("train_features_*.jsonl"))
test_file = data_dir / "test_features.jsonl"

print("Collecting training record hashes...", flush=True)
train_hashes = set()

for path in train_files:
    print(f"Reading {path.name}", flush=True)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            digest = record.get("sha256")
            if digest:
                train_hashes.add(digest.lower())

print(f"Unique training hashes: {len(train_hashes):,}", flush=True)
print("Checking test records...", flush=True)

test_count = 0
overlap_count = 0

with test_file.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        record = json.loads(line)
        test_count += 1
        digest = record.get("sha256")
        if digest and digest.lower() in train_hashes:
            overlap_count += 1

print("\n=== HASH OVERLAP RESULTS ===")
print(f"Test records checked: {test_count:,}")
print(f"Matching hashes:      {overlap_count:,}")
print(f"Overlap percentage:   {100 * overlap_count / test_count:.6f}%")
