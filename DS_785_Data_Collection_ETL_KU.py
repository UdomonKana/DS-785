
import pandas as pd, numpy as np
from datetime import datetime

fcc = pd.read_csv('outputs/source_fcc_notices.csv')
cpsc = pd.read_csv('outputs/source_cpsc_recalls.csv')
etsi = pd.read_csv('outputs/source_etsi_meta.csv')

for df, name in [(fcc,'FCC'),(cpsc,'CPSC'),(etsi,'ETSI')]:
    df['dataset_source'] = name

def normalize_region(val):
    if pd.isna(val): return val
    v = str(val).strip().upper()
    if 'US' in v or 'UNITED STATES' in v: return 'US'
    if 'EU' in v or 'EUROPE' in v: return 'EU'
    if 'CAN' in v: return 'CAN'
    return v

def to_iso_date(s):
    if pd.isna(s): return np.nan
    s = str(s)
    for fmt in ('%Y-%m-%d','%m/%d/%Y','%Y/%m/%d','%d.%m.%Y','%m/%d/%y'):
        try:
            return datetime.strptime(s, fmt).strftime('%Y-%m-%d')
        except: pass
    # fallback for dd.mm.yyyy with 1-digit day/month
    try:
        d,m,y = s.replace('.','-').replace('/','-').split('-')
        return datetime(int(y), int(m), int(d)).strftime('%Y-%m-%d')
    except: return np.nan

for df in [fcc,cpsc,etsi]:
    df['region'] = df['region'].apply(normalize_region)
    df['pub_date_iso'] = df['pub_date'].apply(to_iso_date)
    df['raw_text'] = df['raw_text'].astype(str)
    df['text_length'] = df['raw_text'].str.len()
    df['doc_type'] = df['doc_type'].replace({'Standard Revision':'StandardRevision'})

corpus_cols = ['doc_id','region','doc_type','pub_date_iso','raw_text','text_length',
               'revision_class','product_area','impact_label','dataset_source']
corpus = pd.concat([fcc[corpus_cols], cpsc[corpus_cols], etsi[corpus_cols]], ignore_index=True)
corpus.to_csv('outputs/combined_corpus_raw.csv', index=False)
