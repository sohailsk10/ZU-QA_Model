import pandas as pd
from keybert import KeyBERT
import glob
import os

kw_model = KeyBERT(model='all-mpnet-base-v2')

filename = "remove_404_csv - Copy"
all_csv_ = []
q_tag = 0
csv_files = glob.glob(os.path.join(filename, "*.csv"))
for f in csv_files:
    df = pd.read_csv(f)
    for index, row in df.iterrows():
        try:
            path = row['path']
            try:
                title = row['title']
            except:
                title = row['name']

        except:
            title = row['ServiceName']
            # path = row['GeneratedLink']
            path = row['ServiceUrl']

        all_csv_.append([q_tag, title, path])
        q_tag += 1

NEW_DF_ = pd.DataFrame(all_csv_, columns=['q_tag', 'title', 'path'])
NEW_DF_['bert_keywords'] = None

keywords = []

for index, row in NEW_DF_.iterrows():
    new_df_path = row.path.split("/main")[1]
    # new_df_path = row.path.split(".ae")[1]

# message = 'https://www.zu.ac.ae/main/files/contents/convention_cener/pdf/ZUCC_BrochureEng.pdf'

    keywords = kw_model.extract_keywords(new_df_path.strip().lower(), keyphrase_ngram_range=(1, 7), stop_words='english', highlight=False, top_n=10)
    print(index, len(keywords), f)
    first_elements = [x[0] for x in keywords]
    
    NEW_DF_.at[index, 'bert_keywords'] = ' '.join(first_elements)

NEW_DF_['title'] = NEW_DF_[['bert_keywords', 'title']].apply(lambda x: ' '.join(x), axis=1)

NEW_DF_.to_csv("Getnewservices_with_keywords.csv")