import os
import random
import json
import pickle
import numpy as np
import nltk
  
from keras.models import Sequential
from nltk.stem import WordNetLemmatizer
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import tensorflow as tf  
  
lemmatizer = WordNetLemmatizer()
  
# reading the json.intense file
intents = json.loads(open("data/tf_chatbot/train/intents.json").read())
save_name = 'model/tf_chatbot/zu'  
# creating empty lists to store data
words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # separating words from patterns
        word_list = nltk.word_tokenize(pattern.lower())
        words.extend(word_list)  # and adding them to words list
          
        # associating patterns with respective tags
        documents.append(((word_list), intent["tag"]))
  
        # appending the tags to the class list
        if intent["tag"] not in classes:
            classes.append(intent["tag"])
  
# storing the root words or lemma
words = [lemmatizer.lemmatize(word)
         for word in words if word not in ignore_letters]
words = sorted(set(words))
  
# saving the words and classes list to binary files
pickle.dump(words, open("model/tf_chatbot/zu/words.pkl", "wb"))
pickle.dump(classes, open("model/tf_chatbot/zu/classes.pkl", "wb"))

training = []
output_empty = [0]*len(classes)
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(
        word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
          
    # making a copy of the output_empty
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])
random.shuffle(training)
training = np.array(training)
  
# splitting the data
train_x = list(training[:, 0])
train_y = list(training[:, 1])
input_shape = (len(train_x[0]),)
output_shape = len(train_y[0])

# creating a Sequential machine learning model
model = Sequential()
model.add(Dense(2 * input_shape[0], input_shape=input_shape, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense( input_shape[0], activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(output_shape, activation="softmax"))
  
# compiling the model
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_name + os.sep + "model.{epoch:02d}-{val_loss:.4f}.h5",
                                                                   save_weights_only=False,  monitor="val_loss",   
                                                                   mode="min",    save_best_only=True)

model.compile(loss="categorical_crossentropy",
              optimizer=sgd, metrics=["accuracy"])
hist = model.fit(np.array(train_x), np.array(train_y), epochs=2000, batch_size=5, verbose=1,validation_split=0.2, 
                 shuffle=True,callbacks=[model_checkpoint_callback])

# saving the model
model.save("model/tf_chatbot/zu/chatbotmodel.h5", hist)
  
# print statement to show the 
# successful training of the Chatbot model
print("Yay!")