import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random

lemmatize = WordNetLemmatizer()
words=[]
classes=[]
documents=[]
ignore_words=['?','!']

data_file= open('Animebot.json').read()
Animebot= json.loads(data_file)

for intent in Animebot['intents']:
    for pattern in intent['patterns']:
        #each word is tokenize
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #to the corpus we add the token
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#Sort each words, lower each word, and remove duplicates 
words= [lemmatize.lemmatize(w.lower()) for w in words if w not in ignore_words]
words=sorted(list(set(words)))

#sort classes
classes= sorted(list(set(classes)))

#documents= combination between patterns and Animebot
print(len(documents),"documents")
#classes= Animebot
print(len(classes),"classes",classes)
#words = all words, vocab
print(len(words),"unique lemmatized words", words)
pickle.dump(words, open("words.pkl","wb"))
pickle.dump(classes,open('classes.pkl','wb'))

#create our training data
training=[]
#create an empty array for our output
output_empty=[0]*len(classes)
#training set and using bag of words for each sentences

for doc in documents:
        #init our bag of words algorithm
        bag=[]
        #list of tokenized words for the pattern
        pattern_words= doc[0]
         #lemmatize or sort each word- base word is created in base to represent related words
        pattern_words=[lemmatize.lemmatize(word.lower()) for word in pattern_words]

        #our bag of words array will now assign 1, if word found in current pattern
        for w in words:
                bag.append(1) if w in pattern_words else bag.append(0)
            #output is a 0 for each tag and 1 for each pattern or current tag
                output_row=list(output_empty)
                output_row[classes.index(doc[1])]=1
                training.append([bag, output_row])


#shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

#create train and test lists
train_x=list(training[:,0])
train_y=list(training[:,1])
print("Training data created")

#create model- 3 layers.
#equal to number of Animebot to predict output intent with softmax
model= Sequential()
model.add(Dense(150, input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(len(train_y[0]),activation='softmax'))

#compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model
hist= model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('anime_bot.h5',hist)
print("model is created")