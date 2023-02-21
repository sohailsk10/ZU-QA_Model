from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from keybert import KeyBERT
from numpy import savetxt
from numpy import loadtxt
import glob
import os
from transformers import BertTokenizerFast, BertModel
# , KeyBERTTokenizer
# from transformers import KeyBERTTokenizer


#Define the model. Either from scratch of by loading a pre-trained model
model = SentenceTransformer('bert-base-nli-mean-tokens')
# kw_model = KeyBERT(model=model)
kw_model = KeyBERT(model='all-mpnet-base-v2')

new_df = pd.DataFrame()
all_csv = []

read_csv_folder = "remove_404_csv"
csv_files = glob.glob(os.path.join(read_csv_folder, "*.csv"))
q_tag = 0
for f in csv_files:
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
# Vectorize the questions 
# questions_vec = model.encode(new_df['title'])
questions_vec = loadtxt('question_vec.csv', delimiter=',')
# savetxt('question_vec.csv', questions_vec, delimiter=',')

def cosine_similarity_fn(questions_asked_vec, questions_vec):
	score = []
	dump_score = []
	# for i in range(len(questions)):
	for i in range(len(new_df['title'])):
		cs = cosine_similarity([questions_asked_vec][0], [questions_vec[i]])
		# qtag_value = questions.get('qtag')[i]
		qtag_value = new_df['q_tag'][i]
		current_score = (qtag_value, cs[0][0])
		if len(dump_score) == 0:
			dump_score.append(current_score)
		else:
			for j in range(len(dump_score)):
				if dump_score[j][1] < current_score[1]:
					dump_score.insert(j, current_score)
					dump_score.pop(j+1)
				
		if cs[0][0] > 0.65:
			if len(score) == 0 :
				score.append(current_score)
			else: 
				prev_len = len(score)
				j=0
				while j < len(score):
					if current_score not in score:
						if score[j][1] < current_score[1] and score[j][0]!= current_score[0]:
							score.insert(j, current_score)
						elif score[j][1] < current_score[1] and score[j][0 ]== current_score[0]:
							score.insert(j, current_score)
							score.pop(j+1)
							break
						elif score[j][1] >= current_score[1] and score[j][0]== current_score[0]:
							j+=1
							continue
						elif j == len(score) - 1:
							score.append(current_score)
							break
					else:
						if score[j][1] < current_score[1] and score[j][0] == current_score[0]:
							score.pop(j)
					j+=1
	if len(score) == 0:
		if len(dump_score) == 0:
			return "Sorry, I don't know the answer to that question."
		else:
			score = dump_score
		

	print(score)

	# display results
	ans_list = []
	ans_disp = ""
	tags=[]
	for index, value in score:
		# tag = answers.get('qtag')[index]
		tag = new_df['path'][index]
		if tag not in tags:    
			tags.append(tag)
		ans_disp +=list(new_df.loc[new_df['path']==tag]['path'])[0]+"\n"
		ans_list.append(list(new_df.loc[new_df['path']==tag]['path'])[0])
        # new_df.loc[new_df['path'].isin(ans_disp)]['path']
	
	return ans_disp, ans_list

while True:
	message = input("Type exit to quit, else ask your question\n")
	if message == "exit":
		break
	keywords = kw_model.extract_keywords(message.strip().lower(), keyphrase_ngram_range=(1, 7), stop_words='english', highlight=False, top_n=10)
	# keywords = kw_model.extract_keywords(message.strip().lower(), keyphrase_ngram_range=(1, 7), stop_words=None, highlight=True, top_n=10)
	print('before command dictionary')
	print(keywords)
	keywords_list= list(dict(keywords).keys())
	print("keywords_list[0]", keywords_list[0])
	questions_asked = [keywords_list[0]]
	questions_asked_vec = model.encode(questions_asked)
	# questions_asked_vec = loadtxt('questions_asked_vec.csv', delimiter=',')
	res, res_list = cosine_similarity_fn(questions_asked_vec, questions_vec)
	# res = set(res)
	print("#1", len(res_list))
	res_list = list(set(res_list))
	print("#2", len(res_list))
	print(res, res_list)