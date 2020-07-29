
# coding: utf-8

# In[1]:



# Libraries needed for NLP
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


# Libraries needed for Tensorflow processing

import tensorflow as tf
import numpy as np
import tflearn
import random
import json
import pickle

import pandas as pd
import csv
from sklearn.model_selection import train_test_split
import Recommenders


# In[3]:


song_df_1=pd.read_csv('user_data.csv')
song_df_2=pd.read_csv('song_data.csv')
song_df = pd.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")
song_grouped = song_df.groupby(['title']).agg({'listen_count': 'count'}).reset_index()
grouped_sum = song_grouped['listen_count'].sum()
song_grouped['percentage']  = song_grouped['listen_count'].div(grouped_sum)*100
song_grouped.sort_values(['listen_count', 'title'], ascending = [0,1])
train_data, test_data = train_test_split(song_df, test_size = 0.20, random_state=0)


# In[4]:



# import our chat-bot intents file
with open('intents.json') as json_data:
    intents = json.load(json_data)


# In[5]:


words = []
classes = []
documents = []
ignore = ['?']
# loop through each sentence in the intent's patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each and every word in the sentence
        w = nltk.word_tokenize(pattern)
        # add word to the words list
        words.extend(w)
        # add word(s) to documents
        documents.append((w, intent['tag']))
        # add tags to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


# In[6]:


# Perform stemming and lower each word as well as remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore]
words = sorted(list(set(words)))

# remove duplicate classes
classes = sorted(list(set(classes)))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)


# In[7]:


# create training data
training = []
output = []
# create an empty array for output
output_empty = [0] * len(classes)

# create training set, bag of words for each sentence
for doc in documents:
    # initialize bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stemming each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is '1' for current tag and '0' for rest of other tags
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffling features and turning it into np.array
random.shuffle(training)
training = np.array(training)

# creating training lists
train_x = list(training[:,0])
train_y = list(training[:,1])


# In[8]:


# resetting underlying graph data
tf.reset_default_graph()

# Building neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Defining model and setting up tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

# Start training
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')


# In[9]:


pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )


# In[10]:


# restoring all the data structures
data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']


# In[11]:


#with open('intents.json') as json_data:
#    intents = json.load(json_data)


# In[12]:


# load the saved model
model.load('./model.tflearn')


# In[13]:


def clean_up_sentence(sentence):
    # tokenizing the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stemming each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# returning bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenizing the pattern
    sentence_words = clean_up_sentence(sentence)
    # generating bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))


# In[14]:


context={}
ERROR_THRESHOLD = 0.30
def classify(sentence):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list



def response(sentence, userID='b80344d063b5ccb3212f76538f3d9e43d87dca9e', show_details=False):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                     # set context for this intent if necessary
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details:
                            print ('tag:', i['tag'])
                        if(i['tag']=="genremthriller" or i['tag']=="genremromance" or i['tag']=="genremcomedy" or i['tag']=="genrebthriller" or  i['tag']=="genrebromance" or i['tag']=="genrebcomedy" or i['tag']=="genrespop" or i['tag']=="genresrock" or i['tag']=="genreshiphop"):
                            
                            pm = Recommenders.popularity_recommender_py()
                            rec=pm.create(train_data, 'user_id', 'title')
                            user_id = userID
                            rec=list(rec)
                            return print("Companionbot:",random.choice(i['responses']),"\n",*rec , sep='\n')
                            return print("Companionbot:",random.choice(i['responses']))
                        else:
                            return print("Companionbot:",random.choice(i['responses']))

            results.pop(0)

'''Testing
classify('Suggest me a movie')
classify('Hi')
context
response('Hi there!',show_details=True)
response('Suggest me a movie',show_details=True)
classify('Thriller')
response('Thriller',show_details=True)
response('Thriller')
response('suggest me a song')
response('Pop')
response('suggest me a book')
response('Thriller')
response("I want to read a book")
response("Thriller")
sentence='Hello'
classify("I want movie book")
'''
