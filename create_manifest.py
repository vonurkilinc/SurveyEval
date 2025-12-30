#!/usr/bin/env python3
"""
Create step3_ok_manifest.csv from image_records.jsonl
"""
import json
import csv
from pathlib import Path
from collections import defaultdict

# Load records
records = []
with open('features/image_records.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

# Load feature keys to check what's available
import numpy as np

geom_keys = set()
clip_keys = set()
id_keys = set()

try:
    geom_data = np.load('features/geometry_descriptors.npz')
    geom_keys = set(geom_data['keys'])
except:
    pass

try:
    clip_data = np.load('features/clip_embeddings.npz')
    clip_keys = set(clip_data['keys'])
except:
    pass

try:
    id_data = np.load('features/identity_embeddings.npz')
    id_keys = set(id_data['keys'])
except:
    pass

# Create manifest
manifest_rows = []

for rec in records:
    key = rec['key']
    person_id = rec.get('person_id', '').lower()
    method = rec.get('method', '').upper()
    image_rel = rec.get('image_rel', '').replace('\\', '/')
    reference_rel = rec.get('reference_rel', '').replace('\\', '/')
    
    # Format paths to start with "img/" and use lowercase person_id in paths
    # Convert person_id to lowercase for paths (e.g., P000 -> p000)
    person_id_lower = person_id.lower()
    
    # Extract path parts and reconstruct with lowercase person_id
    image_parts = image_rel.replace('\\', '/').split('/')
    if len(image_parts) > 0 and image_parts[0].upper() == person_id.upper():
        image_parts[0] = person_id_lower
    image_path = f"img/{'/'.join(image_parts)}"
    
    reference_parts = reference_rel.replace('\\', '/').split('/')
    if len(reference_parts) > 0 and reference_parts[0].upper() == person_id.upper():
        reference_parts[0] = person_id_lower
    reference_path = f"img/{'/'.join(reference_parts)}"
    
    # Check which features are OK
    geometry_ok = key in geom_keys
    clip_ok = key in clip_keys
    identity_ok = key in id_keys
    
    # Only include caricatures (skip reference images as separate entries)
    if rec.get('role') == 'caricature':
        manifest_rows.append({
            'person_id': person_id,
            'method': method,
            'image_path': image_path,
            'reference_path': reference_path,
            'geometry_ok': geometry_ok,
            'clip_ok': clip_ok,
            'identity_ok': identity_ok
        })

# Sort by person_id, then method, then image_path
manifest_rows.sort(key=lambda x: (x['person_id'], x['method'], x['image_path']))

# Create reports directory
reports_dir = Path('reports')
reports_dir.mkdir(exist_ok=True)

# Write CSV
output_path = reports_dir / 'step3_ok_manifest.csv'
with open(output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['person_id', 'method', 'image_path', 'reference_path', 'geometry_ok', 'clip_ok', 'identity_ok'])
    writer.writeheader()
    for row in manifest_rows:
        # Convert booleans to Python bool strings (True/False)
        row['geometry_ok'] = str(row['geometry_ok'])
        row['clip_ok'] = str(row['clip_ok'])
        row['identity_ok'] = str(row['identity_ok'])
        writer.writerow(row)

print(f"Created manifest: {output_path}")
print(f"Total rows: {len(manifest_rows)}")
print(f"\nFirst 5 rows:")
for i, row in enumerate(manifest_rows[:5], 1):
    print(f"  {i}. {row}")

