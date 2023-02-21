from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from keybert import KeyBERT
#Define the model. Either from scratch of by loading a pre-trained model
model = SentenceTransformer('bert-base-nli-mean-tokens')
kw_model = KeyBERT(model='all-mpnet-base-v2')
# Datasets to be in the following format:

questions = pd.DataFrame({"qtag":[0,1,2,3,3,4,5,6,7,8,9],
						  "questions":["contact IT support", 
						 "timings happiness center IT support",
						 "convention center fact sheet pdf", 
						 "convention center brochure english pdf ZUCC_BrochureEng.pdf  convention_cener pdf zucc_brochureeng pdf zucc_brochureeng pdf",
						 "request lab support",
						 "support evening classes master working hours",
						 "help fix technical issues home",
						 "divert extension mobile phone",
						 "arrange conduct meetings working remotely",
						 "zu eparticipation policy"]
						})
answers = pd.DataFrame({
						"qtag":[0,1,2,3,4,5,6,7,8,9],
						"answers":["IT Support can be contacted through following channels\nRegister your IT requests or issues by logging to the portal \n              https://sanad.zu.ac.ae\nSend an email to IT.ServiceDesk@zu.ac.e\nContact through phone at 02 599 3666 for Abu Dhabi and 04 402 1777 for Dubai\n", 
						"Happiness Center is open for support from 8 am to 6 pm.",
						"https://www.zu.ac.ae/main/files/contents/convention_cener/pdf/FACT%20SHEET.pdf",
						"https://www.zu.ac.ae/main/files/contents/convention_cener/pdf/ZUCC_BrochureEng.pdf",
						"To request for lab support, please get in touch with the respective lab support personnel as mentioned below:\nDubai Campus\nCACE Lab : Marcus Tolledo\nCCMS Lab : Fathima AlHammadi\nCTI : Emerson Bautista\nAbu Dhabi Campus \nCACE Lab : Ayesh Ghanim\nCCMS Lab : Amar Moraje\nCTI : Fatima AlKarbi\n",
						"Support for Master classes is provided on request basis, which can be registered in SANAD using the portal https://sanad.zu.ac.ae.\nThe request has to be registered at least one week in advance.\n",
						"Yes, we can assist you through remote support. You need to just open a request in SANAD through https://sanad.zu.ac.ae or send an email to IT.ServiceDesk@zu.ac.ae\n\n",
						"Please raise a service request in SANAD as detailed in point 5 aboveSANAD and Choose -> I need a new service -> to request for transferring extension to mobile phone.",
						"ZU has licenses for Webex, ZOOM and Microsoft Teams collaboration software.\nZOOM is primarily used for academic purposes, Webex for administrative and Microsoft Teams as a backup.\nA request can be raised to provide access to any of the collaboration tools to arrange and conduct meetings. The only requirement for these tools to be provisioned is you should have a ZU ID number. \n",
						"https://www.zu.ac.ae/main/en/e-participation/e-participation-policy.aspx"
						]
						})
# Vectorize the questions 
questions_vec = model.encode(questions.get('questions'))

def cosine_similarity_fn(questions_asked_vec, questions_vec):
	score = []
	dump_score = []
	for i in range(len(questions)):
		cs = cosine_similarity(
			[questions_asked_vec[0]], [questions_vec[i]]
			)
		qtag_value = questions.get('qtag')[i]
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
	ans_disp = ""
	tags=[]
	for index, value in score:
		tag = answers.get('qtag')[index]
		if tag not in tags:    
			tags.append(tag)
		ans_disp +=list(answers.loc[answers['qtag']==tag]['answers'])[0]+"\n"
	
	return ans_disp

while True:
	message = input("Type exit to quit, else ask your question\n")
	if message == "exit":
		break
	keywords = kw_model.extract_keywords(message.strip().lower(), keyphrase_ngram_range=(1, 7), stop_words='english', highlight=False, top_n=10)
	print('before command dictionary')
	print(keywords)
	keywords_list= list(dict(keywords).keys())
	print(keywords_list[0])
	questions_asked = [keywords_list[0]]
	questions_asked_vec = model.encode(questions_asked)
	res = cosine_similarity_fn(questions_asked_vec, questions_vec)
	print(res)