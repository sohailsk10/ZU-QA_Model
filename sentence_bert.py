from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
from numpy import savetxt
from numpy import loadtxt

model = SentenceTransformer('bert-base-nli-mean-tokens')


questions_asked = ["Show the map for convention center in pdf"]

questions = pd.DataFrame({"qtag":[0,1,2,3,4,5,6],
                          "questions":["contact IT support", 
                         "timings happiness center IT support",
                         "convention center fact sheet", 
                         "convention center brochure pdf",
                         'show map convention center pdf',
                         'animation design course zu',
                         "pull list library services"]
                        })
# print(questions)
answers = pd.DataFrame({
    "qtag":[0,1,2,3,4,5,6],
    "answers":["IT Support can be contacted through following channels\nRegister your IT requests or issues by logging to the portal \n              https://sanad.zu.ac.ae\nSend an email to IT.ServiceDesk@zu.ac.e\nContact through phone at 02 599 3666 for Abu Dhabi and 04 402 1777 for Dubai\n", 
    "Happiness Center is open for support from 8 am to 6 pm.",
    "https://www.zu.ac.ae/main/files/contents/convention_cener/pdf/FACT%20SHEET.pdf",
    "https://www.zu.ac.ae/main/files/contents/convention_cener/pdf/ZUCC_BrochureEng.pdf",
    "https://www.zu.ac.ae/main/files/contents/convention_cener/pdf/Map_en.pdf",
    "https://www.zu.ac.ae/main/en/colleges/colleges/__college_of_arts_and_creative_enterprises/Academic_programs/Animation.aspx",
    "https://www.zu.ac.ae/main/en/library/services.aspx"
]
})
# print(answers)

# questions_vec = model.encode(questions.get('questions'))
# savetxt('questions_vec.csv', questions_vec, delimiter=',')
# questions_asked_vec = model.encode(questions_asked)
# savetxt('questions_asked_vec.csv', questions_asked_vec, delimiter=',')

# print(questions_vec, type(questions_vec))
# print(questions_asked_vec, type(questions_asked_vec))

questions_asked_vec =loadtxt('questions_asked_vec.csv', delimiter=',')
questions_vec =loadtxt('questions_vec.csv', delimiter=',')
score = []
for i in range(len(questions)):
    # cs = cosine_similarity(
    #     [questions_asked_vec[0]], [questions_vec[i]]
    #     )
    cs = cosine_similarity([questions_asked_vec], [questions_vec[i]]
        )
    qtag_value = questions.get('qtag')[i]
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
print(score)

ans_disp = []
for index, value in score:
    tag = answers.get('qtag')[index]

    if tag not in ans_disp:    
        ans_disp.append(tag)
print(answers.loc[answers['qtag'].isin(ans_disp)]['answers'])

with open('ans_disp.text', 'w') as f:
    f.writelines(str(ans_disp))
# print(answers)