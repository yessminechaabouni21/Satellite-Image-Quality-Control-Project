#!/usr/bin/env python3
"""
check_esa_mismatch.py -- Cross-check ESA scenes between feature table and database.
"""

import pandas as pd
import sqlite3

# Check what's in the feature table for ESA scenes
df = pd.read_csv('reports/ml_features.csv')
esa = df[df['source'] == 'esa_ref']

print('=== ESA rows in feature table ===')
print(f'Total: {len(esa)}')
print(f'label==1 count: {(esa["label"]==1).sum()}')
print()

print('Columns related to defect:')
for col in ['label', 'failed_indicator', 'defect_type', 'defect_family']:
    if col in esa.columns:
        print(f'  {col}: {esa[col].value_counts().to_dict()}')
    else:
        print(f'  {col}: NOT IN TABLE')
print()

# Cross-check with DB
conn = sqlite3.connect('reports/eo_qc.db')
db = pd.read_sql('''
    SELECT scene_name, esa_flag, failed_indicator
    FROM esa_reference
    WHERE esa_flag = "FAILED"
''', conn)
conn.close()

print('=== ESA-FAILED scenes in DB ===')
print(db[['scene_name', 'esa_flag', 'failed_indicator']].to_string())
print()

# Check if scene names match between feature table and DB
esa_names = set(esa[esa['label'] == 1]['scene_name'])
db_names = set(db['scene_name'])

print(f'FAILED in feature table: {len(esa_names)}')
print(f'FAILED in DB: {len(db_names)}')
print(f'Intersection: {len(esa_names & db_names)}')

if esa_names - db_names:
    print(f'\nIn feature table but not DB: {esa_names - db_names}')
if db_names - esa_names:
    print(f'\nIn DB but not feature table: {db_names - esa_names}')
