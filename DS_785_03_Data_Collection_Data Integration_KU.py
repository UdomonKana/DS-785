# Harmonize keys and derive year; keep provenance (dataset_source).
# Inputs
#     outputs/combined_corpus_raw.csv
# Outputs
#     outputs/combined_corpus_integrated.csv

import pandas as pd

corpus = pd.read_csv('outputs/combined_corpus_raw.csv')
corpus['year'] = pd.to_datetime(corpus['pub_date_iso']).dt.year
corpus['product_area'] = corpus['product_area'].fillna('unknown')
corpus.to_csv('outputs/combined_corpus_integrated.csv', index=False)




