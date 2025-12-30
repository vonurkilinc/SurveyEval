#!/usr/bin/env python3
import csv

rows = list(csv.DictReader(open('reports/step3_ok_manifest.csv')))

all_ok = [r for r in rows if r['geometry_ok'] == 'True' and r['clip_ok'] == 'True' and r['identity_ok'] == 'True']

print("=" * 70)
print("FEATURE EXTRACTION STATUS SUMMARY")
print("=" * 70)
print(f"\nTotal images: {len(rows)}")
print(f"\nAll features OK (geometry + CLIP + identity): {len(all_ok)} ({len(all_ok)/len(rows)*100:.1f}%)")

print(f"\nIndividual Feature Success Rates:")
print(f"  Geometry OK: {sum(1 for r in rows if r['geometry_ok'] == 'True')} ({sum(1 for r in rows if r['geometry_ok'] == 'True')/len(rows)*100:.1f}%)")
print(f"  CLIP OK: {sum(1 for r in rows if r['clip_ok'] == 'True')} ({sum(1 for r in rows if r['clip_ok'] == 'True')/len(rows)*100:.1f}%)")
print(f"  Identity OK: {sum(1 for r in rows if r['identity_ok'] == 'True')} ({sum(1 for r in rows if r['identity_ok'] == 'True')/len(rows)*100:.1f}%)")

print(f"\nFeature Combinations:")
print(f"  ✓ Geometry + CLIP + Identity: {len(all_ok)}")
print(f"  ✓ Geometry + CLIP (no Identity): {sum(1 for r in rows if r['geometry_ok'] == 'True' and r['clip_ok'] == 'True' and r['identity_ok'] == 'False')}")
print(f"  ✓ CLIP + Identity (no Geometry): {sum(1 for r in rows if r['geometry_ok'] == 'False' and r['clip_ok'] == 'True' and r['identity_ok'] == 'True')}")
print(f"  ✓ CLIP only: {sum(1 for r in rows if r['geometry_ok'] == 'False' and r['clip_ok'] == 'True' and r['identity_ok'] == 'False')}")
print(f"  ✗ Other combinations: {len(rows) - len(all_ok) - sum(1 for r in rows if r['geometry_ok'] == 'True' and r['clip_ok'] == 'True' and r['identity_ok'] == 'False') - sum(1 for r in rows if r['geometry_ok'] == 'False' and r['clip_ok'] == 'True' and r['identity_ok'] == 'True') - sum(1 for r in rows if r['geometry_ok'] == 'False' and r['clip_ok'] == 'True' and r['identity_ok'] == 'False')}")

print(f"\nBy Method (all features OK):")
methods = {}
for r in rows:
    method = r['method']
    if method not in methods:
        methods[method] = {'total': 0, 'all_ok': 0}
    methods[method]['total'] += 1
    if r['geometry_ok'] == 'True' and r['clip_ok'] == 'True' and r['identity_ok'] == 'True':
        methods[method]['all_ok'] += 1

for method in sorted(methods.keys()):
    stats = methods[method]
    pct = stats['all_ok'] / stats['total'] * 100
    print(f"  {method}: {stats['all_ok']}/{stats['total']} ({pct:.1f}%)")

print("=" * 70)

