from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
import sys
import glob
import os
from numpy import savetxt
from numpy import loadtxt

model = SentenceTransformer('bert-base-nli-mean-tokens')


questions_asked = []

# df = pd.read_csv('GetServices_new.csv')
new_df = pd.DataFrame()
all_csv = []

read_csv_folder = "remove_404_csv"
csv_files = glob.glob(os.path.join(read_csv_folder, "*.csv"))
q_tag = 0
for f in csv_files:
    print(f)
    df = pd.read_csv(f)
    for index, row in df.iterrows():
        try:
            path = row['path']
            # q_tag = row['q_tag']
            try:
                title = row['title']
            except:
                title = row['name']
            try:
                created_on = row['created-on']
            except:
                created_on = row['created_on']
        except:
            title = row['ServiceName']
            path = row['GeneratedLink']

        all_csv.append([q_tag, title, path, created_on])
        # print(q_tag, index, title, path)
        q_tag+=1

new_df =  pd.DataFrame(all_csv, columns=['q_tag', 'title', 'path', 'timestamp'])
# print(new_df.shape)
# questions_vec = model.encode(new_df['title'])
# savetxt('questions_vec.csv', questions_vec, delimiter=',')
questions_vec = loadtxt('questions_vec.csv', delimiter=',')

while True:
    text = input("Enter Your Question: ")
    questions_asked.append(text)
    # questions_asked_vec = model.encode(questions_asked)
    # savetxt('questions_asked_vec.csv', questions_vec, delimiter=',')
    questions_asked_vec = loadtxt('questions_asked_vec.csv', delimiter=',')

    score = []
    for i in range(len(new_df['title'])):
        cs = cosine_similarity([questions_asked_vec], [questions_vec[i]])
        qtag_value = new_df['q_tag'][i]
        current_score = (qtag_value, cs[0][0])
        if cs[0][0] > 0.7:
            if len(score) == 0 :
                score.append(current_score)
            else: 
                prev_len = len(score)
                for j in range(len(score)):
                    if score[j][1] < current_score[1] and score[j][0]!= current_score[0]:
                        score.insert(j, current_score)
                    elif score[j][1] < current_score[1] and score[j][0]== current_score[0]:
                        score.insert(j, current_score)
                        score.pop(j+1)
                    elif score[j][1] >= current_score[1] and score[j][0]== current_score[0]:
                        continue
                    else:
                        score.append(current_score)
                        break
    print("score", score)

    ans_disp = []
    for index, value in score:
        tag = new_df['path'][index]
        if tag not in ans_disp:    
            ans_disp.append(tag)

    print("ans_disp", ans_disp)
    # print(df['path'].loc[df['qtag'].isin(ans_disp)]['path'])
    # print("##############", df.loc[df['GeneratedLink'].isin(ans_disp)]['GeneratedLink'].to_string())
    get_ans = new_df.loc[new_df['path'].isin(ans_disp)]['path']
    get_ans.to_csv("hdhd.csv")
    print(get_ans, type(get_ans))
    questions_asked.clear()
# print(answers.loc[answers['qtag'].isin(ans_disp)]['answers'])
# print(answers)