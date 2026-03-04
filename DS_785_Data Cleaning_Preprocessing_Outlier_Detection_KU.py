
import json, pandas as pd, matplotlib.pyplot as plt

corpus = pd.read_csv('outputs/combined_after_formatting.csv')

before = len(corpus)
corpus = corpus.drop_duplicates(subset=['doc_id'], keep='first')
after = len(corpus)

q05, q95 = corpus['text_length'].quantile([0.05, 0.95])
corpus['is_text_outlier'] = (corpus['text_length'] < q05) | (corpus['text_length'] > q95)

with open('outputs/outliers_report.json','w') as f:
    json.dump({'rows_before':before, 'rows_after':after,
               'q05_text_length':float(q05), 'q95_text_length':float(q95),
               'n_text_outliers':int(corpus['is_text_outlier'].sum())}, f, indent=2)

plt.hist(corpus['text_length'], bins=10, color='#5e35b1')
plt.title('Text Length Histogram'); plt.xlabel('Text Length (chars)'); plt.ylabel('Count')
plt.tight_layout(); plt.savefig('outputs/text_length_hist.png', dpi=160); plt.close()

corpus.to_csv('outputs/combined_after_outlier_handling.csv', index=False)
