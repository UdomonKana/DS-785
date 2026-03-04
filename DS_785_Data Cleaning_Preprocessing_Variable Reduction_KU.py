
import pandas as pd

corpus = pd.read_csv('outputs/combined_after_outlier_handling.csv')
corpus['source_url'] = 'https://example.org/placeholder'
corpus['html_style'] = '<div style=\"color:red\">'
reduced = corpus.drop(columns=['html_style','source_url'])

pd.DataFrame({'dropped_variables':['html_style','source_url']}).to_csv('outputs/variable_reduction_dropped.csv', index=False)
reduced.to_csv('outputs/combined_after_variable_reduction.csv', index=False)
