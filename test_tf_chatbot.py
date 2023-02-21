import random
import json
import pickle
import numpy as np
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("data/tf_chatbot/train/intents.json").read())
words = pickle.load(open('model/tf_chatbot/zu/words.pkl', 'rb'))
classes = pickle.load(open('model/tf_chatbot/zu/classes.pkl', 'rb'))
model = load_model('model/tf_chatbot/zu/chatbotmodel.h5')

def clean_up_sentences(sentence):
	sentence_words = nltk.word_tokenize(sentence.lower())
	sentence_words = [lemmatizer.lemmatize(word)
					for word in sentence_words]
	return sentence_words

def bagw(sentence):
	sentence_words = clean_up_sentences(sentence)
	bag = [0]*len(words)
	for w in sentence_words:
		for i, word in enumerate(words):
			if word == w:
				bag[i] = 1
	return np.array(bag)

def predict_class(sentence):
	bow = bagw(sentence)
	res = model.predict(np.array([bow]))[0]
	ERROR_THRESHOLD = 0.25
	results = [[i, r] for i, r in enumerate(res)
			if r > ERROR_THRESHOLD]
	results.sort(key=lambda x: x[1], reverse=True)
	return_list = []
	print_list = []
	for r in results:
		print_list.append({'intent': classes[r[0]],
							'probability': str(r[1])})
	print(print_list)
	for r in results:
		return_list.append({'intent': classes[r[0]],
                            'probability': str(r[1])})
	return return_list

def get_response(intents_list, intents_json):
	tags = [ x['intent'] for x in intents_list]
	list_of_intents = intents_json['intents']
	result = ""
	for i in list_of_intents:
		if i['tag'] in tags:
			result+=random.choice(i['responses']) + "\n"
			# break
			# break
	return result

print("Chatbot is up!")

while True:
	message = input("")
	ints = predict_class(message)
	res = get_response(ints, intents)
	print(res)
