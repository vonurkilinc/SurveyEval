#!/usr/bin/env python3
import json

with open('features/image_records.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}\n")

clip_warnings = []
for line in lines:
    rec = json.loads(line)
    warnings = rec.get('status', {}).get('warnings', [])
    if any('CLIP encoder unavailable' in w for w in warnings):
        clip_warnings.append(rec)

print(f"Records with CLIP warnings: {len(clip_warnings)}\n")
print("Sample records with CLIP warnings:")
for r in clip_warnings[:5]:
    print(f"  Key: {r['key']}")
    print(f"  Image: {r.get('image_rel', 'N/A')}")
    print(f"  Timestamp: {r.get('timestamp', 'N/A')}")
    print(f"  Warnings: {r.get('status', {}).get('warnings', [])}")
    print()

print("\nLast 10 records:")
for r in [json.loads(l) for l in lines[-10:]]:
    print(f"  Key: {r['key']}, Image: {r.get('image_rel', 'N/A')[:40]}")
    print(f"    Timestamp: {r.get('timestamp', 'N/A')}")
    print(f"    Status OK: {r.get('status', {}).get('ok', False)}")
    print(f"    Warnings: {r.get('status', {}).get('warnings', [])}")
    print()

