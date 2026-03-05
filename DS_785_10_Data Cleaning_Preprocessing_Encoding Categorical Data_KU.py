# One‑hot encoding for region, doc_type, and revision_class; ordinal encoding for product_area.
#Inputs
#      outputs/combined_after_scaling.csv
# Outputs
#      outputs/combined_after_encoding.csv

import pandas as pd

df = pd.read_csv('outputs/combined_after_scaling.csv')

prod_order = {'hardware':3, 'software':2, 'manual':1, 'labeling':0, 'packaging':1}
df['product_area_ord'] = df['product_area'].map(prod_order).fillna(1).astype(int)

encoded = pd.get_dummies(df, columns=['region','doc_type','revision_class'],
                         prefix=['region','doctype','rev'])
encoded.to_csv('outputs/combined_after_encoding.csv', index=False)

