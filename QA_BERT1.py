# from transformers import BertForQuestionAnswering
# from transformers import AutoTokenizer
# from transformers import pipeline
# import os
import json
# import torch
# from pytorch_pretrained_bert import BertModel
# from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
# from transformers import BertConfig
# from transformers import BertForSequenceClassification
# from transformers import BertForTokenClassification
# model = BertForQuestionAnswering.from_pretrained('distilbert-base-uncased')
# tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased') 
# # with open('./data/bert_train_set/train-v2.0.json','rb') as f:
#     squad_dict = json.load(f)


contexts = []

contexts.append("""IT Support can be contacted through following channels
Register your IT requests or issues by logging to the portal 
              https://sanad.zu.ac.ae
Send an email to IT.ServiceDesk@zu.ac.e
Contact through phone at 02 599 3666 for Abu Dhabi and 04 402 1777 for Dubai
""")

contexts.append("""Happiness Center is open for support from 8 am – 6 pm.""")
contexts.append("""To request for lab support, please get in touch with the respective lab support personnel as mentioned below:
Dubai Campus
CACE Lab – Marcus Tolledo
CCMS Lab – Fathima AlHammadi
CTI – Emerson Bautista
Abu Dhabi Campus 
CACE Lab – Ayesh Ghanim
CCMS Lab – Amar Moraje
CTI – Fatima AlKarbi
""")
contexts.append("""Support for Master classes is provided on request basis, which can be registered in SANAD using the portal https://sanad.zu.ac.ae.
The request has to be registered at least one week in advance.
""")
contexts.append("""Yes, we can assist you through remote support. You need to just open a request in SANAD through https://sanad.zu.ac.ae or send an email to IT.ServiceDesk@zu.ac.ae
""")
contexts.append("""Please raise a service request in SANAD as detailed in point 5 aboveSANAD and Choose “ I need a new service” to request for transferring extension to mobile phone.
""")
contexts.append("""ZU has licenses for Webex, ZOOM and Microsoft Teams collaboration software.
ZOOM is primarily used for academic purposes, Webex for administrative and Microsoft Teams as a backup.
A request can be raised to provide access to any of the collaboration tools to arrange and conduct meetings. The only requirement for these tools to be provisioned is you should have a ZU ID number. 
""")

questions=[]
questions.append("How can I contact IT Support?")
questions.append("Timings of happiness center for IT support")
questions.append("How do I request lab support? ")
questions.append("Support for evening classes i.e Master or after working hours ")
questions.append("Can you help me fix technical issues at home?")
questions.append("How do I divert my extension to my mobile phone?")
questions.append("How do I arrange & conduct meetings when I am working remotely?")
contexts_str = ''
# for context in contexts:
#     contexts_str+= context
# print(contexts)
# print(questions)
# tokenizer.encode(questions[0], truncation=True, padding=True)
# nlp = pipeline('question-answering',model,tokenizer=tokenizer)
# # print(tokenized_q1)
# result = nlp({
#     'question': questions[1],
#     'context': contexts_str           
# })

# print(result)

train_json = {"data":[{"title":"IT support & general questions", "paragraphs":[],}]}

answers = []
train_json["data"][0]['paragraphs'].append({"qas":[]})
# print(train_json["data"][0]['paragraphs'])
for i in range(len(contexts)):
    contexts_str+= contexts[i]
    answer_start = contexts_str.find(contexts[i])
    print("answer_start", answer_start)
    answer_end = answer_start + len(contexts[i])
    if i < len(contexts)-1:
        data_dict_paragraphs={
            "question": questions[i], "id": i, "answers":[{"text": contexts[i], "answer_start": answer_start, "answer_end":answer_end}], "is_impossible": False
            }
    else:
        data_dict_paragraphs={
            "question": questions[i], "id": i, "answers":[{"text": contexts[i], "answer_start": answer_start, "answer_end":answer_end}], "is_impossible": False
        }
    answers.append({"text": contexts[i], "answer_start": answer_start, "answer_end":answer_end})
    train_json["data"][0]['paragraphs'][0]['qas'].append(data_dict_paragraphs) 
train_json["data"][0]['paragraphs'].append({'context': contexts_str})
# print(train_json)
with open('context.txt', 'w') as f:
    f.writelines(contexts_str)
    f.close()


# contexts_list = [contexts_str] * len(contexts)
# print(answers[6])
# with open('data/bert_train_set/train.json', 'w') as jsonfile:
#     json.dump(train_json, jsonfile)
# # print(train_json)
# train_encodings = tokenizer(contexts_list, questions, truncation=True, padding=True)
# print(train_encodings.keys())
# # print(tokenizer.decode(train_encodings['input_ids'][0]))
# print(answers[0]['answer_start'])
# train_encodings.char_to_token(0, answers[6]['answer_end'])  
# print(answers[6]['answer_end'])
# # train_encodings.char_to_token(0, answers[1]['answer_start'])
# start_positions = []
# end_positions = []
# for i in range(len(contexts)):
#     start_positions.append(train_encodings.char_to_token(i,answers[i]['answer_start']))
#     end_positions.append(train_encodings.char_to_token(i,answers[i]['answer_end']))
#     j=1
#     while end_positions[-1] is None:
#         end_positions[-1] = train_encodings.char_to_token(i,answers[i]['answer_end']-j)         
#         j+=1


# train_encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
# print(train_encodings['end_positions'][:100])
# class SquadDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings):
#         self.encodings = encodings
#     def __getitem__(self, idx):
#         return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#     def __len__(self):
#         return len(self.encodings.input_ids)
    
# train_datasets = SquadDataset(train_encodings)

# from torch.utils.data import DataLoader
# from torch import optim as optimz
# from tqdm import tqdm

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# model.train()
# optim = optimz.AdamW(model.parameters(), lr=5e-5)
# train_loader = DataLoader(train_datasets, batch_size=16, shuffle=True )
# for epoch in range(3):
#     loop = tqdm(train_loader)
#     for batch in loop:
#         optim.zero_grad()
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         start_positions = batch['start_positions'].to(device)
#         end_positions = batch['end_positions'].to(device)
#         outputs = model(input_ids = input_ids, attention_mask = attention_mask, start_positions = start_positions, end_positions = end_positions)
#         loss = outputs[0]
#         loss.backward()
#         optim.step()
#         loop.set_description(f'Epoch{epoch}')
#         loop.set_postfix(loss=loss.item())


# model_path = 'model/zuqa_distilbert'
# model.save_pretrained(model_path)
# tokenizer.save_pretrained(model_path)
# nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

# nlp({'question': questions[3], 'context': contexts_list[0]})