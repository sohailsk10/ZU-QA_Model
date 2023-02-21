import pandas as pd
from keybert import KeyBERT
import glob
import os

kw_model = KeyBERT(model='all-mpnet-base-v2')

filename = "remove_404_csv"
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
            path = row['GeneratedLink']

        all_csv_.append([q_tag, title, path])
        q_tag += 1

NEW_DF = pd.DataFrame(all_csv_, columns=['q_tag', 'title', 'path'])

keywords = []

for index, row in NEW_DF.iterrows():
    new_df_path = row.path

# message = 'https://www.zu.ac.ae/main/files/contents/convention_cener/pdf/ZUCC_BrochureEng.pdf'

    keywords = kw_model.extract_keywords(new_df_path.strip().lower(), keyphrase_ngram_range=(1, 7), stop_words='english', highlight=False, top_n=10)
    first_elements = [x[0] for x in keywords]
    first_elements = pd.Series(first_elements)
    # print(first_elements)
    
        # keywords.append(i)
    NEW_DF['bert_keywords'] = first_elements
    
    

print(NEW_DF.head())

NEW_DF.to_csv('final_data.csv')
# keywords.clear()
        
    
# for i in first_elements:
    
# print(first_elements)
# print(keywords)
# keywords_list= list(dict(keywords).keys())
# print()
# print("###",keywords_list[0], keywords_list[0][1])