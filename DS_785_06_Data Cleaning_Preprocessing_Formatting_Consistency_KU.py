
# 6) Data Cleaning & Preprocessing – Formatting & Consistency (Self-contained & resilient, with directory creation)
import os, re, json, unicodedata
import pandas as pd, numpy as np
from datetime import datetime

os.makedirs('outputs', exist_ok=True)

PRIMARY_INPUT  = 'outputs/combined_after_missing_handling.csv'
FALLBACK_INPUT = 'outputs/combined_corpus_integrated.csv'
OUTPUT_PATH    = 'outputs/combined_after_formatting.csv'
BEFORE_AFTER   = 'outputs/formatting_before_after_examples.csv'
SUMMARY_PATH   = 'outputs/formatting_summary.json'

# Load or synthesize input
if os.path.exists(PRIMARY_INPUT):
    df = pd.read_csv(PRIMARY_INPUT)
    source_used = PRIMARY_INPUT
elif os.path.exists(FALLBACK_INPUT):
    df = pd.read_csv(FALLBACK_INPUT)
    source_used = FALLBACK_INPUT
else:
    data = [
        {
            'doc_id':'FCC-2022-050','region':'United States','doc_type':'Rulemaking',
            'pub_date':'01/20/2022','pub_date_iso':'01/20/2022',
            'raw_text':'<p>Labeling clarification for small devices.</p>',
            'revision_class':'labeling','product_area':'labels',
            'impact_label':'labeling'
        },
        {
            'doc_id':'ETSI-EN-301-489','region':'EU','doc_type':'Standard Revision',
            'pub_date':'5.12.2021','pub_date_iso':'5.12.2021',
            'raw_text':'EMC standard update referencing radio modules.',
            'revision_class':'technical','product_area':'Hardware',
            'impact_label':'hardware'
        }
    ]
    df = pd.DataFrame(data)
    source_used = 'synthetic_inline'

# Helpers
def normalize_region(val):
    if pd.isna(val): return np.nan
    v = str(val).strip().upper()
    if any(tok in v for tok in ['UNITED STATES','USA','US']): return 'US'
    if any(tok in v for tok in ['EUROPE','EU']): return 'EU'
    if any(tok in v for tok in ['CANADA','CAN']): return 'CAN'
    return v

KNOWN_FORMATS = ('%Y-%m-%d','%m/%d/%Y','%Y/%m/%d','%d.%m.%Y','%m/%d/%y','%d-%m-%Y','%d-%m-%y')
def to_iso_date(s):
    if pd.isna(s): return np.nan
    s = str(s).strip()
    for fmt in KNOWN_FORMATS:
        try:
            return datetime.strptime(s, fmt).strftime('%Y-%m-%d')
        except: pass
    try:
        d,m,y = s.replace('.','-').replace('/','-').split('-')
        return datetime(int(y), int(m), int(d)).strftime('%Y-%m-%d')
    except: return np.nan

PRODUCT_MAP = {'hardware':'hardware','manual':'manual','labeling':'labeling',
               'labelling':'labeling','labels':'labeling','packaging':'packaging'}
def normalize_product_area(v):
    if pd.isna(v) or str(v).strip()=='' : return 'manual'
    return PRODUCT_MAP.get(str(v).strip().lower(), str(v).strip().lower())

TAG_RE = re.compile(r'<[^>]+>')
def clean_text(s):
    s = '' if pd.isna(s) else str(s)
    s = TAG_RE.sub('', s)
    s = unicodedata.normalize('NFC', s)
    return s.strip()

# Before snapshot for example table
old_date = df['pub_date_iso'].iloc[0] if 'pub_date_iso' in df.columns else (df['pub_date'].iloc[0] if 'pub_date' in df.columns else '')
old_region = df['region'].iloc[0] if 'region' in df.columns else ''
old_area   = df['product_area'].iloc[0] if 'product_area' in df.columns else ''

# Apply formatting & consistency
if 'pub_date_iso' not in df.columns:
    if 'pub_date' in df.columns: df['pub_date_iso'] = df['pub_date'].apply(to_iso_date)
    else: df['pub_date_iso'] = np.nan
else:
    df['pub_date_iso'] = df['pub_date_iso'].apply(to_iso_date)

if 'region' in df.columns: df['region'] = df['region'].apply(normalize_region)
else: df['region'] = np.nan

if 'product_area' in df.columns: df['product_area'] = df['product_area'].apply(normalize_product_area)
else: df['product_area'] = 'manual'

DOC_TYPES = {'Rulemaking':'Rulemaking','Recall':'Recall','Standard Revision':'StandardRevision',
             'StandardRevision':'StandardRevision','Enforcement':'Enforcement'}
df['doc_type'] = df['doc_type'].map(lambda x: DOC_TYPES.get(str(x), str(x))) if 'doc_type' in df.columns else 'Rulemaking'

if 'raw_text' in df.columns: df['raw_text'] = df['raw_text'].apply(clean_text)
else: df['raw_text'] = ''

_dt = pd.to_datetime(df['pub_date_iso'], errors='coerce')
df['year'] = _dt.dt.year.astype('Int64')

if 'text_length' in df.columns:
    df['text_length'] = pd.to_numeric(df['text_length'], errors='coerce').astype('Int64')
else:
    df['text_length'] = df['raw_text'].str.len().astype('Int64')

# After snapshot & example table
new_date = df['pub_date_iso'].iloc[0]
new_region = df['region'].iloc[0]
new_area = df['product_area'].iloc[0]

ex = pd.DataFrame({'Field':['Date Example','Region Example','Product Area Example'],
                   'Before':[old_date, old_region, old_area],
                   'After':[new_date, new_region, new_area]})
ex.to_csv(BEFORE_AFTER, index=False)

summary = {
    'source_used': source_used,
    'n_rows': int(df.shape[0]),
    'n_cols': int(df.shape[1]),
    'regions': df['region'].value_counts(dropna=False).to_dict(),
    'doc_types': df['doc_type'].value_counts(dropna=False).to_dict() if 'doc_type' in df.columns else {},
    'product_area_vocab': sorted(df['product_area'].dropna().unique().tolist()),
    'missing_by_col': df.isna().sum().to_dict()
}
with open(SUMMARY_PATH, 'w') as f:
    json.dump(summary, f, indent=2)

# Save final
df.to_csv(OUTPUT_PATH, index=False)
print('Formatting & Consistency complete.')
print('Input used:', source_used)
print('Wrote:', OUTPUT_PATH)
print('Wrote:', BEFORE_AFTER)
print('Wrote:', SUMMARY_PATH)

