
import pandas as pd

rows = [
    {'category':'Regulatory','source_name':'FCC','url':'https://www.fcc.gov',
     'access_method':'public website / HTML & PDF',
     'license_note':'Public regulatory information; no redistribution of full standards texts.'},
    {'category':'Regulatory','source_name':'ISED Canada','url':'https://ised-isde.canada.ca',
     'access_method':'public website','license_note':'Public regulatory information.'},
    {'category':'Regulatory','source_name':'EUR-Lex','url':'https://eur-lex.europa.eu',
     'access_method':'public website / HTML & PDF','license_note':'EU legal publications for transparency.'},
    {'category':'Safety/Recalls','source_name':'CPSC','url':'https://www.cpsc.gov/Recalls',
     'access_method':'CSV/JSON download','license_note':'Public recall database.'},
    {'category':'Safety/Recalls','source_name':'EU Safety Gate (RAPEX)','url':'https://ec.europa.eu/safety-gate',
     'access_method':'web portal / JSON','license_note':'Public alerts.'},
    {'category':'Safety/Recalls','source_name':'Health Canada Recalls','url':'https://recalls-rappels.canada.ca',
     'access_method':'web portal','license_note':'Public alerts.'},
    {'category':'Standards','source_name':'ETSI','url':'https://www.etsi.org/standards',
     'access_method':'public website','license_note':'Metadata freely accessible; full text redistribution restricted.'},
    {'category':'Standards','source_name':'ISO/IEC metadata','url':'https://www.iso.org/standards.html',
     'access_method':'public website','license_note':'Public metadata; full text paywalled; no redistribution.'},
]
cat = pd.DataFrame(rows)
cat.to_csv('outputs/data_sources_catalog.csv', index=False)
