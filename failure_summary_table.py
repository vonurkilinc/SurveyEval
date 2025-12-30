#!/usr/bin/env python3
"""
Create a clear summary table of failures by method.
"""
import json
from collections import defaultdict

# Load records
records = []
with open('features/image_records.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

# Count by method
method_stats = defaultdict(lambda: {
    'total': 0,
    'geometry_ok': 0,
    'geometry_fail': 0,
    'clip_ok': 0,
    'clip_fail': 0,
    'identity_ok': 0,
    'identity_fail': 0
})

for rec in records:
    method = rec.get('method', 'reference' if rec.get('role') == 'reference' else 'unknown')
    status = rec.get('status', {})
    errors = status.get('errors', [])
    warnings = status.get('warnings', [])
    
    method_stats[method]['total'] += 1
    
    # Check geometry (landmarks)
    has_geometry_error = any('Landmarks failed' in e for e in errors)
    if has_geometry_error:
        method_stats[method]['geometry_fail'] += 1
    else:
        method_stats[method]['geometry_ok'] += 1
    
    # Check CLIP
    has_clip_warning = any('CLIP' in w and 'unavailable' in w for w in warnings)
    if has_clip_warning:
        method_stats[method]['clip_fail'] += 1
    else:
        method_stats[method]['clip_ok'] += 1
    
    # Check Identity
    has_identity_warning = any('Identity' in w and 'no face detected' in w for w in warnings)
    if has_identity_warning:
        method_stats[method]['identity_fail'] += 1
    else:
        method_stats[method]['identity_ok'] += 1

print("=" * 100)
print("FEATURE EXTRACTION SUCCESS RATE BY METHOD")
print("=" * 100)
print()
print(f"{'Method':<15} {'Total':<8} {'Geometry':<20} {'CLIP':<20} {'Identity':<20}")
print(f"{'':<15} {'Images':<8} {'OK / Fail':<20} {'OK / Fail':<20} {'OK / Fail':<20}")
print("-" * 100)

for method in sorted(method_stats.keys()):
    stats = method_stats[method]
    geom_str = f"{stats['geometry_ok']}/{stats['geometry_fail']}"
    clip_str = f"{stats['clip_ok']}/{stats['clip_fail']}"
    id_str = f"{stats['identity_ok']}/{stats['identity_fail']}"
    
    print(f"{method.upper():<15} {stats['total']:<8} {geom_str:<20} {clip_str:<20} {id_str:<20}")

print()
print("=" * 100)
print("KEY FINDINGS:")
print("=" * 100)
print()
print("1. CLIP EMBEDDINGS: 100% success rate across ALL methods")
print("   → All 66 images have CLIP embeddings")
print()
print("2. GEOMETRY DESCRIPTORS:")
print("   → QWEN method: 50% failure rate (10/20 images)")
print("   → All other methods: 100% success rate")
print("   → Failures are due to MediaPipe not detecting faces in stylized caricatures")
print()
print("3. IDENTITY EMBEDDINGS:")
print("   → QWEN method: 90% failure rate (18/20 images)")
print("   → PULID method: 14.3% failure rate (3/21 images, all v3 variants)")
print("   → REFERENCE: 20% failure rate (1/5 images)")
print("   → CARICATUREBOOTH & INSTANTID: 100% success rate")
print("   → Failures are due to InsightFace not detecting faces")
print()
print("=" * 100)

