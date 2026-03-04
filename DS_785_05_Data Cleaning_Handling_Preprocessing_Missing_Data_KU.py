# Impute product_area from domain patterns in text (no synthetic labels).
# Inputs
#     outputs/combined_corpus_integrated.csv
# Outputs
#     outputs/combined_after_missing_handling.csv

import pandas as pd

corpus = pd.read_csv('outputs/combined_corpus_integrated.csv')

kw_map = {'tractor|loader|gnss|radio|module|overheat':'hardware',
          'label|manual':'labeling'}

def infer_product_area(text):
    t = str(text).lower()
    for patt, val in kw_map.items():
        for tok in patt.split('|'):
            if tok in t: return val
    return 'manual'  # fallback

mask = (corpus['product_area'] == 'unknown')
corpus.loc[mask, 'product_area'] = corpus.loc[mask, 'raw_text'].apply(infer_product_area)

corpus.to_csv('outputs/combined_after_missing_handling.csv', index=False)

