
import json, pandas as pd, matplotlib.pyplot as plt

corpus = pd.read_csv('outputs/combined_corpus_integrated.csv')

profile = {
    'n_rows': int(corpus.shape[0]),
    'n_cols': int(corpus.shape[1]),
    'regions': corpus['region'].value_counts().to_dict(),
    'doc_types': corpus['doc_type'].value_counts().to_dict(),
    'missing_by_col': corpus.isna().sum().to_dict(),
    'impact_distribution': corpus['impact_label'].value_counts().to_dict(),
    'text_length_summary': corpus['text_length'].describe().to_dict(),
}
with open('outputs/profile_summary.json','w') as f:
    json.dump(profile, f, indent=2)

corpus['impact_label'].value_counts().plot(kind='bar', color=['#2e7d32','#0277bd','#6a1b9a'])
plt.title('Impact Label Distribution (All)')
plt.xlabel('Impact Label'); plt.ylabel('Count'); plt.tight_layout()
plt.savefig('outputs/impact_distribution_all.png', dpi=160); plt.close()
