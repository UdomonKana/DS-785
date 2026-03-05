#  z‑score standardization for numeric metadata (e.g., text_length, year).
# Inputs
#       outputs/combined_after_variable_reduction.csv
# Outputs
#       outputs/scaled_numeric_features.csv
#       outputs/combined_after_scaling.csv

# 9) Data Cleaning & Preprocessing – Scaling & Normalization (Complete & Robust)
# Reads prior outputs, scales numeric features with multiple strategies, and writes plots & summaries.

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

os.makedirs('outputs', exist_ok=True)

CANDIDATE_INPUTS = [
    'outputs/combined_after_variable_reduction.csv',
    'outputs/combined_after_outlier_handling.csv',
    'outputs/combined_after_formatting.csv',
    'outputs/combined_corpus_integrated.csv'
]

# ---------- Load or synthesize input ----------
for p in CANDIDATE_INPUTS:
    if os.path.exists(p):
        df = pd.read_csv(p)
        source_used = p
        break
else:
    # synthesize a tiny frame with realistic columns
    df = pd.DataFrame({
        'doc_id':['FCC-2022-050','CPSC-2021-TRAC-01','ETSI-EN-301-489'],
        'year':[2022, 2021, 2021],
        'text_length':[68, 112, 94],
        'impact_label':['labeling','labeling','hardware'],
        'product_area':['labeling','labeling','hardware'],
        'raw_text':['Labeling clarification','Recall notice text','EMC standard update'],
    })
    source_used = 'synthetic_inline'

# ---------- Identify numeric candidates ----------
# Keep a small, explicit list and intersect with existing columns for safety.
numeric_candidates = ['text_length','year']
num_cols = [c for c in numeric_candidates if c in df.columns]

# If nothing numeric is present, synthesize a numeric column to demonstrate
if not num_cols:
    df['text_length'] = df['raw_text'].astype(str).str.len()
    num_cols = ['text_length']

# ---------- Simple distributions BEFORE ----------
plt.figure(figsize=(6,3))
for i, c in enumerate(num_cols, 1):
    plt.subplot(1, len(num_cols), i)
    plt.hist(df[c].dropna(), bins=10, color='#546e7a', alpha=0.9)
    plt.title(f'{c} (pre)')
    plt.tight_layout()
plt.savefig('outputs/scaling_hist_pre.png', dpi=160)
plt.close()

# ---------- Transformations ----------
scaled = df.copy()

# 1) Z-score standardization (for models assuming Gaussian-ish features)
if num_cols:
    z_scaler = StandardScaler()
    scaled[[f'{c}_z' for c in num_cols]] = z_scaler.fit_transform(scaled[num_cols])

# 2) Min-Max scaling to [0,1] (for distance-based or bounded feature needs)
if num_cols:
    mm_scaler = MinMaxScaler()
    scaled[[f'{c}_minmax' for c in num_cols]] = mm_scaler.fit_transform(scaled[num_cols])

# 3) Robust scaling (resistant to outliers)
if num_cols:
    rb_scaler = RobustScaler()
    scaled[[f'{c}_robust' for c in num_cols]] = rb_scaler.fit_transform(scaled[num_cols])

# 4) Optional log scaling for positive, skewed features (guard against <=0)
for c in num_cols:
    if (scaled[c] > 0).all():
        scaled[f'{c}_log'] = np.log1p(scaled[c])

# ---------- Simple distributions AFTER (z-scored only for visual compactness) ----------
plt.figure(figsize=(6,3))
for i, c in enumerate(num_cols, 1):
    colz = f'{c}_z'
    if colz in scaled.columns:
        plt.subplot(1, len(num_cols), i)
        plt.hist(scaled[colz].dropna(), bins=10, color='#1e88e5', alpha=0.9)
        plt.title(f'{c}_z (post)')
        plt.tight_layout()
plt.savefig('outputs/scaling_hist_post.png', dpi=160)
plt.close()

# ---------- Save artifacts ----------
scaled.to_csv('outputs/combined_after_scaling.csv', index=False)
scaled[['doc_id'] + [c for c in scaled.columns if c.endswith('_z') or c.endswith('_minmax') or c.endswith('_robust') or c.endswith('_log')]]\
      .to_csv('outputs/scaled_feature_matrix.csv', index=False)

summary = {
    'source_used': source_used,
    'numeric_columns_used': num_cols,
    'created_columns': [c for c in scaled.columns if c.endswith('_z') or c.endswith('_minmax') or c.endswith('_robust') or c.endswith('_log')],
    'n_rows': int(scaled.shape[0])
}
with open('outputs/scaling_normalization_summary.json','w') as f:
    json.dump(summary, f, indent=2)

print('Scaling & Normalization complete.')
print('Input used:', source_used)
print('Numeric columns:', num_cols)
print('Wrote: outputs/combined_after_scaling.csv')
print('Wrote: outputs/scaled_feature_matrix.csv')
print('Wrote: outputs/scaling_hist_pre.png, outputs/scaling_hist_post.png')
print('Wrote: outputs/scaling_normalization_summary.json')

