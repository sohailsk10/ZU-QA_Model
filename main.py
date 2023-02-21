# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import json
import string
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout
import os
from tensorflow.keras.callbacks import EarlyStopping

save_name = 'model/tf/zu01'
# used a dictionary to represent an intents JSON file
data = {"intents": [
    {"tag": "Brochure",
     "patterns": ['convention center brochure english pdf', 'convention center brochure pdf english', 'convention center english brochure pdf', 'convention center english pdf brochure', 'convention center pdf brochure english', 'convention center pdf english brochure', 'convention brochure center english pdf', 'convention brochure center pdf english', 'convention brochure english center pdf', 'convention brochure english pdf center', 'convention brochure pdf center english', 'convention brochure pdf english center', 'convention english center brochure pdf', 'convention english center pdf brochure', 'convention english brochure center pdf', 'convention english brochure pdf center', 'convention english pdf center brochure', 'convention english pdf brochure center', 'convention pdf center brochure english', 'convention pdf center english brochure', 'convention pdf brochure center english', 'convention pdf brochure english center', 'convention pdf english center brochure', 'convention pdf english brochure center', 'center convention brochure english pdf', 'center convention brochure pdf english', 'center convention english brochure pdf', 'center convention english pdf brochure', 'center convention pdf brochure english', 'center convention pdf english brochure', 'center brochure convention english pdf', 'center brochure convention pdf english', 'center brochure english convention pdf', 'center brochure english pdf convention', 'center brochure pdf convention english', 'center brochure pdf english convention', 'center english convention brochure pdf', 'center english convention pdf brochure', 'center english brochure convention pdf', 'center english brochure pdf convention', 'center english pdf convention brochure', 'center english pdf brochure convention', 'center pdf convention brochure english', 'center pdf convention english brochure', 'center pdf brochure convention english', 'center pdf brochure english convention', 'center pdf english convention brochure', 'center pdf english brochure convention', 'brochure convention center english pdf', 'brochure convention center pdf english', 'brochure convention english center pdf', 'brochure convention english pdf center', 'brochure convention pdf center english', 'brochure convention pdf english center', 'brochure center convention english pdf', 'brochure center convention pdf english', 'brochure center english convention pdf', 'brochure center english pdf convention', 'brochure center pdf convention english', 'brochure center pdf english convention', 'brochure english convention center pdf', 'brochure english convention pdf center', 'brochure english center convention pdf', 'brochure english center pdf convention', 'brochure english pdf convention center', 'brochure english pdf center convention', 'brochure pdf convention center english', 'brochure pdf convention english center', 'brochure pdf center convention english', 'brochure pdf center english convention', 'brochure pdf english convention center', 'brochure pdf english center convention', 'english convention center brochure pdf', 'english convention center pdf brochure', 'english convention brochure center pdf', 'english convention brochure pdf center', 'english convention pdf center brochure', 'english convention pdf brochure center', 'english center convention brochure pdf', 'english center convention pdf brochure', 'english center brochure convention pdf', 'english center brochure pdf convention', 'english center pdf convention brochure', 'english center pdf brochure convention', 'english brochure convention center pdf', 'english brochure convention pdf center', 'english brochure center convention pdf', 'english brochure center pdf convention', 'english brochure pdf convention center', 'english brochure pdf center convention', 'english pdf convention center brochure', 'english pdf convention brochure center', 'english pdf center convention brochure', 'english pdf center brochure convention', 'english pdf brochure convention center', 'english pdf brochure center convention', 
'pdf convention center brochure english', 'pdf convention center english brochure', 'pdf convention brochure center english', 'pdf convention brochure english center', 'pdf convention english center brochure', 'pdf convention english brochure center', 'pdf center convention brochure english', 'pdf center convention english brochure', 'pdf center brochure convention english', 'pdf center brochure english convention', 'pdf center english convention brochure', 'pdf center english brochure convention', 'pdf brochure convention center english', 'pdf brochure convention english center', 'pdf brochure center convention english', 'pdf brochure center english convention', 'pdf brochure english convention center', 'pdf brochure english center convention', 'pdf english convention center brochure', 'pdf english convention brochure center', 'pdf english center convention brochure', 'pdf english center brochure convention', 'pdf english brochure convention center', 'pdf english brochure center convention', 'convention center brochure english', 'convention center brochure pdf', 'convention center english brochure', 'convention center english pdf', 'convention center pdf brochure', 'convention center pdf english', 'convention brochure center english', 'convention brochure center pdf', 'convention brochure english center', 'convention brochure english pdf', 'convention brochure pdf center', 'convention brochure pdf english', 'convention english center brochure', 'convention english center pdf', 'convention english brochure center', 'convention english brochure pdf', 'convention english pdf center', 'convention english pdf brochure', 'convention pdf center brochure', 'convention pdf center english', 'convention pdf brochure center', 'convention pdf brochure english', 'convention pdf english center', 'convention pdf english brochure', 'center convention brochure english', 'center convention brochure pdf', 'center convention english brochure', 'center convention english pdf', 'center convention pdf brochure', 'center convention pdf english', 'center brochure convention english', 'center brochure convention pdf', 'center brochure english convention', 'center brochure english pdf', 'center brochure pdf convention', 'center brochure pdf english', 'center english convention brochure', 'center english convention pdf', 'center english brochure convention', 'center english brochure pdf', 'center english pdf convention', 'center english pdf brochure', 'center pdf convention brochure', 'center pdf convention english', 'center pdf brochure convention', 'center pdf brochure english', 'center pdf english convention', 'center pdf english brochure', 'brochure convention center english', 'brochure convention center pdf', 'brochure convention english center', 'brochure convention english pdf', 'brochure convention pdf center', 'brochure convention pdf english', 'brochure center convention english', 'brochure center convention pdf', 'brochure center english convention', 'brochure center english pdf', 'brochure center pdf convention', 'brochure center pdf english', 'brochure english convention center', 'brochure english convention pdf', 'brochure english center convention', 'brochure english center pdf', 'brochure english pdf convention', 'brochure english pdf center', 'brochure pdf convention center', 'brochure pdf convention english', 'brochure pdf center convention', 'brochure pdf center english', 'brochure pdf english convention', 'brochure pdf english center', 'english convention center brochure', 'english convention center pdf', 'english convention brochure center', 'english convention brochure pdf', 'english convention pdf center', 'english convention pdf brochure', 'english center convention brochure', 'english center convention pdf', 'english center brochure convention', 'english center brochure pdf', 'english center pdf convention', 'english center pdf brochure', 'english brochure convention center', 'english brochure convention pdf', 'english brochure center convention', 'english brochure center pdf', 'english brochure pdf convention', 'english brochure pdf center', 'english pdf convention center', 'english pdf convention brochure', 'english pdf center convention', 'english pdf center brochure', 'english pdf brochure convention', 'english pdf brochure center', 'pdf convention center brochure', 'pdf convention center english', 'pdf convention brochure center', 'pdf convention brochure english', 'pdf convention english center', 'pdf convention english brochure', 'pdf center convention brochure', 'pdf center convention english', 'pdf center brochure convention', 'pdf center brochure english', 'pdf center english convention', 'pdf center english brochure', 'pdf brochure convention center', 'pdf brochure convention english', 'pdf brochure center convention', 'pdf brochure center english', 'pdf brochure english convention', 'pdf brochure english center', 'pdf english convention center', 'pdf english convention brochure', 'pdf english center convention', 'pdf english center brochure', 'pdf english brochure convention', 'pdf english brochure center', 'convention center brochure', 'convention center english', 'convention center pdf', 'convention brochure center', 'convention brochure english', 'convention brochure pdf', 'convention english center', 'convention english brochure', 'convention english pdf', 'convention pdf center', 'convention pdf brochure', 'convention pdf english', 'center convention brochure', 'center convention english', 'center convention pdf', 'center brochure convention', 'center brochure english', 'center brochure pdf', 'center english convention', 'center english brochure', 'center english pdf', 'center pdf convention', 'center pdf brochure', 'center pdf english', 'brochure convention center', 'brochure convention english', 'brochure convention pdf', 'brochure center convention', 'brochure center english', 'brochure center pdf', 'brochure english convention', 'brochure english center', 'brochure english pdf', 'brochure pdf convention', 'brochure pdf center', 'brochure pdf english', 'english convention center', 'english convention brochure', 'english convention pdf', 'english center convention', 'english center brochure', 'english center pdf', 'english brochure convention', 'english brochure center', 'english brochure pdf', 'english pdf convention', 'english pdf center', 'english pdf brochure', 'pdf convention center', 'pdf convention brochure', 'pdf convention english', 'pdf center convention', 'pdf center brochure', 'pdf center english', 'pdf brochure convention', 'pdf brochure center', 'pdf brochure english', 'pdf english convention', 'pdf english center', 'pdf english brochure', 'convention center', 'convention brochure', 'convention english', 'convention pdf', 'center convention', 'center brochure', 'center english', 'center pdf', 'brochure convention', 'brochure center', 'brochure english', 'brochure pdf', 'english convention', 'english center', 'english brochure', 'english pdf', 'pdf convention', 'pdf center', 'pdf brochure', 'pdf english', 'convention', 'center', 'brochure', 'english', 'pdf'],
     "responses": ["https://www.zu.ac.ae/main/files/contents/convention_cener/pdf/ZUCC_BrochureEng.pdf"]

     },
    {"tag": "Fact_Sheet",
     "patterns": ['convention center factsheet pdf', 'convention center pdf factsheet', 'convention factsheet center pdf', 'convention factsheet pdf center', 'convention pdf center factsheet', 'convention pdf factsheet center', 'center convention factsheet pdf', 'center convention pdf factsheet', 'center factsheet convention pdf', 'center factsheet pdf convention', 'center pdf convention factsheet', 'center pdf factsheet convention', 'factsheet convention center pdf', 'factsheet convention pdf center', 'factsheet center convention pdf', 'factsheet center pdf convention', 'factsheet pdf convention center', 'factsheet pdf center convention', 'pdf convention center factsheet', 'pdf convention factsheet center', 'pdf center convention factsheet', 'pdf center factsheet convention', 'pdf factsheet convention center', 'pdf factsheet center convention', 'convention center factsheet', 'convention center pdf', 'convention factsheet center', 'convention factsheet pdf', 'convention pdf center', 'convention pdf factsheet', 'center convention factsheet', 'center convention pdf', 'center factsheet convention', 'center factsheet pdf', 'center pdf convention', 'center pdf factsheet', 'factsheet convention center', 'factsheet convention pdf', 'factsheet center convention', 'factsheet center pdf', 'factsheet pdf convention', 'factsheet pdf center', 'pdf convention center', 'pdf convention factsheet', 'pdf center convention', 'pdf center factsheet', 'pdf factsheet convention', 'pdf factsheet center', 'convention center', 'convention factsheet', 'convention pdf', 'center convention', 'center factsheet', 'center pdf', 'factsheet convention', 'factsheet center', 'factsheet pdf', 'pdf convention', 'pdf center', 'pdf factsheet', 'convention', 'center', 'factsheet', 'pdf'],
     "responses": ["https://www.zu.ac.ae/main/files/contents/convention_cener/pdf/FACT%20SHEET.pdf"]

     },
     {"tag": "ZU_Catalog ",
     "patterns": ['download zu catalog 21 22 pdf', 'download zu catalog 21 pdf 22', 'download zu catalog 22 21 pdf', 'download zu catalog 22 pdf 21', 'download catalog 21 pdf zu 22', 'download catalog 21 pdf 22 zu', 'download catalog 22 zu 21 pdf', 'download catalog 22 zu pdf 21', 'download catalog 22 21 zu pdf', 'download catalog 22 21 pdf zu', 'download catalog 22 pdf zu 21', 'download catalog 22 pdf 21 zu', 'download catalog pdf zu 21 22', 'download catalog pdf zu 22 21', 'download catalog pdf 21 zu 22', 'download catalog pdf 21 22 zu', 'download catalog pdf 22 zu 21', 'download catalog pdf 22 21 zu'],
     "responses": ["https://www.zu.ac.ae/main/files/contents/edu/Catalog_21_22_ver1.pdf"]

     },
    {"tag": "FMA_conference",
     "patterns": ['fma conference pdf', 'fma pdf conference', 'conference fma pdf', 'conference pdf fma', 'pdf fma conference', 'pdf conference fma', 'fma conference', 'fma pdf', 'conference fma', 'conference pdf', 'pdf fma', 'pdf conference', 'fma', 'conference', 'pdf'],
     "responses": ["https://www.zu.ac.ae/main/files/contents/cbs/docs/CFP-FMA_2022.pdf"]

     },
     {"tag": " map_convention_center",
     "patterns": ['show map convention center pdf', 'show map convention pdf center', 'show map center convention pdf', 'show map center pdf convention', 'show map pdf convention center', 'show map pdf center convention', 'show convention map center pdf', 'show convention map pdf center', 'show convention center map pdf', 'show convention center pdf map', 'show convention pdf map center', 'show convention pdf center map', 'show center map convention pdf', 'show center map pdf convention', 'show center convention map pdf', 'show center convention pdf map', 
'show center pdf map convention', 'show center pdf convention map', 'show pdf map convention center', 'show pdf map center convention', 'show pdf convention map center', 'show pdf convention center map', 'show pdf center map convention', 'show pdf center convention map', 'map show convention center pdf', 'map show convention pdf center', 'map show center convention pdf', 'map show center pdf convention', 'map show pdf convention center', 'map show pdf center convention', 'map convention show center pdf', 'map convention show pdf center', 'map convention center show pdf', 'map convention center pdf show', 'map convention pdf show center', 'map convention pdf center show', 'map center show convention pdf', 'map center show pdf convention', 'map center convention show pdf', 'map center convention pdf show', 'map center pdf show convention', 'map center pdf convention show', 'map pdf show convention center', 'map pdf show center convention', 'map pdf convention show center', 'map pdf convention center show', 'map pdf center show convention', 'map pdf center convention show', 'convention show map center pdf', 'convention show map pdf center', 'convention show center map pdf', 'convention show center pdf map', 'convention show pdf map center', 'convention show pdf center map', 'convention map show center pdf', 'convention map show pdf center', 'convention map center show pdf', 'convention map center pdf show', 'convention map pdf show center', 'convention map pdf center show', 'convention center show map pdf', 'convention center show pdf map', 'convention center map show pdf', 'convention center map pdf show', 'convention center pdf show map', 'convention center pdf map show', 'convention pdf show map center', 'convention pdf show center map', 'convention pdf map show center', 'convention pdf map center show', 'convention pdf center show map', 'convention pdf center map show', 'center show map convention pdf', 'center show map pdf convention', 'center show convention map pdf', 'center show convention pdf map', 'center show pdf map convention', 'center show pdf convention map', 'center map show convention pdf', 'center map show pdf convention', 'center map convention show pdf', 'center map convention pdf show', 'center map pdf show convention', 'center map pdf convention show', 'center convention show map pdf', 'center convention show pdf map', 'center convention map show pdf', 'center convention map pdf show', 'center convention pdf show map', 'center convention pdf map show', 'center pdf show map convention', 'center pdf show convention map', 'center pdf map show convention', 'center pdf map convention show', 'center pdf convention show map', 'center pdf convention map show', 'pdf show map convention center', 'pdf show map center convention', 'pdf show convention map center', 'pdf show convention center map', 'pdf show center map convention', 'pdf show center convention map', 'pdf map show convention center', 'pdf map show center convention', 'pdf map convention show center', 'pdf map convention center show', 'pdf map center show convention', 'pdf map center convention show', 'pdf convention show map center', 'pdf convention show center map', 'pdf convention map show center', 'pdf convention map center show', 'pdf convention center show map', 'pdf convention center map show', 'pdf center show map convention', 'pdf center show convention map', 'pdf center map show convention', 'pdf center map convention show', 'pdf center convention show map', 'pdf center convention map show', 'show map convention center', 'show map convention pdf', 'show map center convention', 'show map center pdf', 'show map pdf convention', 'show map pdf center', 'show convention map center', 'show convention map pdf', 'show convention center map', 'show convention center pdf', 'show convention pdf map', 'show convention pdf center', 'show center map convention', 'show center map pdf', 'show center convention map', 'show center convention pdf', 'show center pdf map', 'show center pdf convention', 'show pdf map convention', 'show pdf map center', 'show pdf convention map', 'show pdf convention center', 'show pdf center map', 'show pdf center convention', 'map show convention center', 'map show convention pdf', 'map show center convention', 'map show center pdf', 'map show pdf convention', 'map show pdf center', 'map convention show center', 'map convention show pdf', 'map convention center show', 'map convention center pdf', 'map convention pdf show', 'map convention pdf center', 'map center show convention', 'map center show pdf', 'map center convention show', 'map center convention pdf', 'map center pdf show', 'map center pdf convention', 'map pdf show convention', 'map pdf show center', 'map pdf convention show', 'map pdf convention center', 'map pdf center show', 'map pdf center convention', 'convention show map center', 'convention show map pdf', 'convention show center map', 'convention show center pdf', 'convention show pdf map', 'convention show pdf center', 'convention map show center', 'convention map show pdf', 'convention map center show', 'convention map center pdf', 'convention map pdf show', 'convention map pdf center', 'convention center show map', 'convention center show pdf', 'convention center map show', 'convention center map pdf', 'convention center pdf show', 'convention center pdf map', 'convention pdf show map', 'convention pdf show center', 'convention pdf map show', 'convention pdf map center', 'convention pdf center show', 'convention pdf center map', 'center show map convention', 'center show map pdf', 'center show convention map', 'center show convention pdf', 'center show pdf map', 'center show pdf convention', 'center map show convention', 'center map show pdf', 'center map convention show', 'center map convention pdf', 'center map pdf show', 'center map pdf convention', 'center convention show map', 'center convention show pdf', 'center convention map show', 'center convention map pdf', 
'center convention pdf show', 'center convention pdf map', 'center pdf show map', 'center pdf show convention', 'center pdf map show', 'center pdf map convention', 'center pdf convention show', 'center pdf convention map', 'pdf show map convention', 'pdf show map center', 'pdf show convention map', 'pdf show convention center', 'pdf show center map', 'pdf show center convention', 'pdf map show convention', 'pdf map show center', 'pdf map convention show', 'pdf map convention center', 'pdf map center show', 'pdf map center convention', 'pdf convention show map', 'pdf convention show center', 'pdf convention map show', 'pdf convention map center', 
'pdf convention center show', 'pdf convention center map', 'pdf center show map', 'pdf center show convention', 'pdf center map show', 'pdf center map convention', 'pdf center convention show', 'pdf center convention map', 'show map convention', 'show map center', 'show map pdf', 'show convention map', 'show convention center', 'show convention pdf', 'show center map', 'show center convention', 'show center pdf', 'show pdf map', 'show pdf convention', 'show pdf center', 'map show convention', 'map show center', 'map show pdf', 'map convention show', 'map convention center', 'map convention pdf', 'map center show', 'map center convention', 'map center pdf', 'map pdf show', 'map pdf convention', 'map pdf center', 'convention show map', 'convention show center', 'convention show pdf', 'convention map show', 'convention map center', 'convention map pdf', 'convention center show', 'convention center map', 'convention center pdf', 'convention pdf show', 'convention pdf map', 'convention pdf center', 'center show map', 'center show convention', 'center show pdf', 'center map show', 'center map convention', 'center map pdf', 'center convention show', 'center convention map', 'center convention pdf', 'center pdf show', 'center pdf map', 'center pdf convention', 'pdf show map', 'pdf show convention', 'pdf show center', 'pdf map show', 'pdf map convention', 'pdf map center', 'pdf convention show', 'pdf convention map', 'pdf convention center', 'pdf center show', 'pdf center map', 'pdf center convention', 'show map', 'show convention', 'show center', 'show pdf', 'map show', 'map convention', 'map center', 'map pdf', 'convention show', 'convention map', 'convention center', 'convention pdf', 'center show', 'center map', 'center convention', 'center pdf', 'pdf show', 'pdf map', 'pdf convention', 'pdf center', 'show', 'map', 'convention', 'center', 'pdf'],
     "responses": ["https://www.zu.ac.ae/main/files/contents/convention_cener/pdf/Map_en.pdf"]
     },
     {"tag": "Animation_design",
     "patterns": ['animion design course zu', 'animion design zu course', 'animion course design zu', 'animion course zu design', 'animion zu design course', 'animion zu course design', 'design animion course zu', 'design animion zu course', 'design course animion zu', 'design course zu animion', 'design zu animion course', 'design zu course animion', 'course animion design zu', 'course animion zu design', 'course design animion zu', 'course design zu animion', 'course zu animion design', 'course zu design animion', 'zu animion design course', 'zu animion course design', 'zu design animion course', 'zu design course animion', 'zu course animion design', 'zu course design animion', 'animion design course', 'animion design zu', 'animion course design', 'animion course zu', 'animion zu design', 'animion zu course', 'design animion course', 'design animion zu', 'design course animion', 'design course zu', 'design zu animion', 'design zu course', 'course animion design', 'course animion zu', 'course design animion', 'course design zu', 'course zu animion', 'course zu design', 'zu animion design', 'zu animion course', 'zu design animion', 'zu design course', 'zu course animion', 'zu course design', 'animion design', 'animion course', 'animion zu', 'design animion', 'design course', 'design zu', 'course animion', 'course design', 'course zu', 'zu animion', 'zu design', 'zu course', 'animion', 'design', 'course', 'zu'],
     "responses": ["https://www.zu.ac.ae/main/en/colleges/colleges/__college_of_arts_and_creative_enterprises/Academic_programs/Animation.aspx"]
     },
     {"tag": "library_services",
     "patterns": ['pull list library services', 'pull list services library', 'pull library list services', 'pull library services list', 'pull services list library', 'pull services library list', 'list pull library services', 'list pull services library', 'list library pull services', 'list library services pull', 'list services pull library', 'list services library pull', 'library pull list services', 'library pull services list', 'library list pull services', 'library list services pull', 'library services pull list', 'library services list pull', 'services pull list library', 'services pull library list', 'services list pull library', 'services list library pull', 'services library pull list', 'services library list pull', 'pull list library', 'pull list services', 'pull library list', 'pull library services', 'pull services list', 'pull services library', 'list pull library', 'list pull services', 'list library pull', 'list library services', 'list services pull', 'list services library', 'library pull list', 'library pull services', 'library list pull', 'library list services', 'library services pull', 'library services list', 'services pull list', 'services pull library', 'services list pull', 'services list library', 'services library pull', 'services library list', 'pull list', 'pull library', 
'pull services', 'list pull', 'list library', 'list services', 'library pull', 'library list', 'library services', 'services pull', 'services list', 'services library', 'pull', 'list', 'library', 'services'],
     "responses": ["https://www.zu.ac.ae/main/en/library/services.aspx"]
     }
]}
# Each list to create
words = []
classes = []
doc_X = []
doc_Y = []
train_X = None
train_Y = None
lemmatizer = WordNetLemmatizer()
model = None


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def download_data():
    nltk.download("punkt")
    nltk.download("wordnet")


def lemmatize_data():
    global words
    global classes
    global doc_X
    global doc_Y
    # global lemmatizer
    # Loop through all the intents
    # tokenize each pattern and append tokens to words, the patterns and
    # the associated tag to their associated list
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens)
            doc_X.append(pattern)
            doc_Y.append(intent["tag"])

        # add the tag to the classes if it's not there already
        if intent["tag"] not in classes:
            classes.append(intent["tag"])
    # lemmatize all the words in the vocab and convert them to lowercase
    # if the words don't appear in punctuation
    words_loc = [lemmatizer.lemmatize(word.lower())
                 for word in words if word not in string.punctuation]
    # sorting the vocab and classes in alphabetical order and taking the # set to ensure no duplicates occur
    words = sorted(set(words))
    classes = sorted(set(classes))


def list_training_data():
    # list for training data
    global words
    global classes
    global doc_X
    global doc_Y
    global train_X
    global train_Y
    training = []
    out_empty = [0] * len(classes)

    # creating the bag of words model
    for idx, doc in enumerate(doc_X):
        bow = []
        text = lemmatizer.lemmatize(doc.lower())
        for word in words:
            bow.append(1) if word in text else bow.append(0)
        # mark the index of class that the current pattern is associated
        # to
        output_row = list(out_empty)
        output_row[classes.index(doc_Y[idx])] = 1
        # add the one hot encoded BoW and associated classes to training
        training.append([bow, output_row])
    # shuffle the data and convert it to an array
    random.shuffle(training)
    training = np.array(training, dtype=object)
    # split the features and target labels
    train_X = np.array(list(training[:, 0]))
    train_Y = np.array(list(training[:, 1]))


def train_data():
    global model
    # defining some parameters
    input_shape = (len(train_X[0]),)
    output_shape = len(train_Y[0])
    epochs = 1500
    # the deep learning model
    model = Sequential()
    model.add(Dense(128, input_shape=input_shape, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(output_shape, activation="softmax"))
    adam = tf.keras.optimizers.Adam(learning_rate=0.00001, decay=1e-6)
    metric = 'val_loss'
    early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min')

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_name + os.sep + 'model.{epoch:02d}-{val_loss:.4f}.h5',
     save_weights_only=False,  monitor=metric,   mode='min',    save_best_only=True)

    
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=["accuracy"])
    print(model.summary())
    model.fit(train_X, train_Y, epochs=epochs, batch_size=16, verbose=1, validation_split=0.3, callbacks=[model_checkpoint_callback, early_stop])


def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bow[idx] = 1
    return np.array(bow)


def pred_class(text, vocab, labels):
    bow = bag_of_words(text, vocab)
    result_pred = model.predict(np.array([bow]))[0]
    thresh = 0.2
    y_pred = [[idx, res]
              for idx, res in enumerate(result_pred) if res > thresh]

    y_pred.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in y_pred:
        return_list.append(labels[r[0]])
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result_resp = random.choice(i["responses"])
            break
    return result_resp


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    download_data()
    lemmatize_data()
    list_training_data()
    train_data()

    while True:
        message = input("Please Type your question below \n")
        if message == "exit":
            break
        print(message)
        intents = pred_class(message, words, classes)
        result_chat = get_response(intents, data)
        print(result_chat)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
