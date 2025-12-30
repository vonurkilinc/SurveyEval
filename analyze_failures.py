#!/usr/bin/env python3
"""
Analyze feature extraction failures by method and image type.
"""
import json
from collections import defaultdict
from pathlib import Path

# Load records
records = []
with open('features/image_records.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

# Organize failures
failures_by_method = defaultdict(lambda: {
    'total': 0,
    'geometry_fail': 0,
    'clip_fail': 0,
    'identity_fail': 0,
    'landmark_fail': 0,
    'images': []
})

failures_by_role = defaultdict(lambda: {
    'total': 0,
    'geometry_fail': 0,
    'clip_fail': 0,
    'identity_fail': 0,
    'landmark_fail': 0,
    'images': []
})

# Analyze each record
for rec in records:
    image_rel = rec.get('image_rel', '')
    role = rec.get('role', 'unknown')
    method = rec.get('method', 'reference' if role == 'reference' else 'unknown')
    status = rec.get('status', {})
    errors = status.get('errors', [])
    warnings = status.get('warnings', [])
    
    # Determine failures
    geometry_fail = False
    clip_fail = False
    identity_fail = False
    landmark_fail = False
    
    for err in errors:
        if 'Landmarks failed' in err:
            landmark_fail = True
            geometry_fail = True
    
    for warn in warnings:
        if 'CLIP' in warn and 'unavailable' in warn:
            clip_fail = True
        if 'Identity' in warn and 'no face detected' in warn:
            identity_fail = True
    
    # Track by method
    failures_by_method[method]['total'] += 1
    if geometry_fail:
        failures_by_method[method]['geometry_fail'] += 1
    if clip_fail:
        failures_by_method[method]['clip_fail'] += 1
    if identity_fail:
        failures_by_method[method]['identity_fail'] += 1
    if landmark_fail:
        failures_by_method[method]['landmark_fail'] += 1
    
    if geometry_fail or clip_fail or identity_fail:
        failures_by_method[method]['images'].append({
            'image': image_rel,
            'geometry_fail': geometry_fail,
            'clip_fail': clip_fail,
            'identity_fail': identity_fail,
            'landmark_fail': landmark_fail,
            'errors': errors,
            'warnings': warnings
        })
    
    # Track by role
    failures_by_role[role]['total'] += 1
    if geometry_fail:
        failures_by_role[role]['geometry_fail'] += 1
    if clip_fail:
        failures_by_role[role]['clip_fail'] += 1
    if identity_fail:
        failures_by_role[role]['identity_fail'] += 1
    if landmark_fail:
        failures_by_role[role]['landmark_fail'] += 1
    
    if geometry_fail or clip_fail or identity_fail:
        failures_by_role[role]['images'].append({
            'image': image_rel,
            'method': method,
            'geometry_fail': geometry_fail,
            'clip_fail': clip_fail,
            'identity_fail': identity_fail,
            'landmark_fail': landmark_fail
        })

# Print report
print("=" * 80)
print("FEATURE EXTRACTION FAILURE ANALYSIS")
print("=" * 80)
print()

print("BY METHOD:")
print("-" * 80)
for method in sorted(failures_by_method.keys()):
    stats = failures_by_method[method]
    print(f"\n{method.upper()}:")
    print(f"  Total images: {stats['total']}")
    print(f"  Geometry failures: {stats['geometry_fail']} ({stats['geometry_fail']/stats['total']*100:.1f}%)")
    print(f"  CLIP failures: {stats['clip_fail']} ({stats['clip_fail']/stats['total']*100:.1f}%)")
    print(f"  Identity failures: {stats['identity_fail']} ({stats['identity_fail']/stats['total']*100:.1f}%)")
    print(f"  Landmark failures: {stats['landmark_fail']} ({stats['landmark_fail']/stats['total']*100:.1f}%)")
    
    if stats['images']:
        print(f"\n  Failed images ({len(stats['images'])}):")
        for img_info in stats['images']:
            failures = []
            if img_info['geometry_fail']:
                failures.append('Geometry')
            if img_info['clip_fail']:
                failures.append('CLIP')
            if img_info['identity_fail']:
                failures.append('Identity')
            print(f"    - {img_info['image']}: {', '.join(failures)}")

print("\n" + "=" * 80)
print("BY ROLE (Reference vs Caricature):")
print("-" * 80)
for role in sorted(failures_by_role.keys()):
    stats = failures_by_role[role]
    print(f"\n{role.upper()}:")
    print(f"  Total images: {stats['total']}")
    print(f"  Geometry failures: {stats['geometry_fail']} ({stats['geometry_fail']/stats['total']*100:.1f}%)")
    print(f"  CLIP failures: {stats['clip_fail']} ({stats['clip_fail']/stats['total']*100:.1f}%)")
    print(f"  Identity failures: {stats['identity_fail']} ({stats['identity_fail']/stats['total']*100:.1f}%)")
    print(f"  Landmark failures: {stats['landmark_fail']} ({stats['landmark_fail']/stats['total']*100:.1f}%)")

print("\n" + "=" * 80)
print("DETAILED FAILURE BREAKDOWN:")
print("-" * 80)

# Group by failure type
geometry_failures = []
clip_failures = []
identity_failures = []

for rec in records:
    image_rel = rec.get('image_rel', '')
    method = rec.get('method', 'reference' if rec.get('role') == 'reference' else 'unknown')
    status = rec.get('status', {})
    errors = status.get('errors', [])
    warnings = status.get('warnings', [])
    
    for err in errors:
        if 'Landmarks failed' in err:
            geometry_failures.append({'image': image_rel, 'method': method, 'error': err})
    
    for warn in warnings:
        if 'CLIP' in warn and 'unavailable' in warn:
            clip_failures.append({'image': image_rel, 'method': method, 'warning': warn})
        if 'Identity' in warn and 'no face detected' in warn:
            identity_failures.append({'image': image_rel, 'method': method, 'warning': warn})

print(f"\nGEOMETRY FAILURES ({len(geometry_failures)}):")
for fail in geometry_failures:
    print(f"  [{fail['method']}] {fail['image']}: {fail['error']}")

print(f"\nCLIP FAILURES ({len(clip_failures)}):")
if clip_failures:
    for fail in clip_failures:
        print(f"  [{fail['method']}] {fail['image']}: {fail['warning']}")
else:
    print("  None! All images have CLIP embeddings.")

print(f"\nIDENTITY FAILURES ({len(identity_failures)}):")
for fail in identity_failures:
    print(f"  [{fail['method']}] {fail['image']}: {fail['warning']}")

print("\n" + "=" * 80)

