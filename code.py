#install dependencies

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

#load data
import json
with open('/content/tweets.json') as jfile:
    d=json.load(jfile)
		
#due to computational complexity I'm taking only a part of it
from sklearn.utils import shuffle
data = shuffle(data).reset_index(drop=True)
data=data[:500]

#The email dataset is very messy so we need to extract the necessary features from it.
def parse_raw_message(raw_message):
    lines = raw_message.split('\n')
    email = {}
    message = ''
    keys_to_extract = ['from', 'to']
    for line in lines:
        if ':' not in line:
            message += line.strip()
            email['body'] = message
        else:
            pairs = line.split(':')
            key = pairs[0].lower()
            val = pairs[1].strip()
            if key in keys_to_extract:
                email[key] = val
    return email

def parse_into_emails(messages):
    emails = [parse_raw_message(message) for message in messages]
    body=[]
    to=[]
    From=[]
    for x in emails:
        body.append(x["body"])
        try:
            to.append(x["to"])
        except:
            to.append(" ")
        From.append(x["from"])
    return body,to,From
    
email_df = pd.DataFrame(parse_into_emails(data.message)) #converting list in dataframes
email_df=email_df.T #doing transpose 
email_df.columns=["Body","To","From"] # assigning new column names

#Check if there are any null values and drop them
email_df.drop(email_df.query("Body == '' | To == '' | From == ''").index, inplace=True)

#now clean the data columns 
import nltk
from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer,PorterStemmer
def cleaner(d):
    #tokenization
    token=word_tokenize(str(d).replace("'","").lower())
    #without punct
    without_punc=[ w for w in token if w.isalpha()]
    #without stopwords
    without_stopwords=[ w for w in without_punc if w not in stop_words]
    #lematization
    word_lem=[WordNetLemmatizer().lemmatize(w) for w in without_stopwords]
    #Stemmatization
    #word_stem=[PorterStemmer().stem(w) for w in word_lem]
    return (" ").join(word_lem)
 
email_df["Body"]=email_df["Body"].apply(cleaner)

#count no of unique words and count of all words
words=email_df["Body"].values
word_dataset=[nltk.word_tokenize(w) for w in words]
s=set()
count=0
for x in word_dataset:
    for y in x:
        s.add(y)
        count+=1
print("total no of unique words",len(list(s)))
print("total words",count)


#use term frequency and inverse document frequency to get relation between each words 
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(stop_words='english', max_df=0.50, min_df=2)
X = vect.fit_transform(email_df.Body)

def top_tfidf_feats(row, features, top_n=20):
    topn_ids = np.argsort(row)[::-1][:top_n]
    #print(topn_ids)
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats, columns=['features', 'score'])
    return df
def top_feats_in_doc(X, features, row_id, top_n=25):
    row = np.squeeze(X[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

features = vect.get_feature_names()
print(top_feats_in_doc(X, features, 1, 10))

def top_mean_feats(X, features,grp_ids=None, min_tfidf=0.1, top_n=25):
    if grp_ids:
        D = X[grp_ids].toarray()
    else:
        D = X.toarray()
    D[D < min_tfidf] = 0
    #print(D)
    tfidf_means = np.mean(D, axis=0)
    #print(tfidf_means)
    return top_tfidf_feats(tfidf_means, features, top_n)

print(top_mean_feats(X, features, top_n=10))

#apply the KMenas clustering algorithm
from sklearn.cluster import KMeans
n_clusters = 2
kmeans= KMeans(n_clusters=2, max_iter=100, init='k-means++', n_init=1)
labels=kmeans.fit_predict(X)

print(labels)
label_0=0
label_1=0
for x in labels:
    if(x==0):
        label_0+=1
    else:
        label_1+=1
print("Total no of 0 label",label_0)

def top_feats_per_cluster(X, y, features, min_tfidf=0.1, top_n=52):
    dfs = []
    labels = np.unique(y)
    print(labels)
    for label in labels:
        ids = np.where(y==label) 
        feats_df = top_mean_feats(X, features, ids,    min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs
	
print(top_feats_per_cluster(X,[0,1],features,0.1,20))#important words based on clustering

#now using keras
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
t=Tokenizer()
t.fit_on_texts(email_df["Body"])
encoded_docs = t.texts_to_sequences(email_df["Body"])
word_count = lambda doc: len(word_tokenize(doc))

longest_doc = max(email_df["Body"], key=word_count)
length_longest_doc = len(word_tokenize(longest_doc))
length_longest_doc = 100

from keras.preprocessing.sequence import pad_sequences
padded_docs =pad_sequences(encoded_docs, length_longest_doc, padding='post',truncating="post")

print("Total no of 1 labels",label_1)

import gensim
model = gensim.models.Word2Vec(word_dataset, vector_size=100, window=5, min_count=1, workers=4, sg=0)

model.wv.most_similar("message")

#applying kmeans on embeddings
from sklearn.cluster import KMeans
k_means=KMeans(n_clusters=2,max_iter=100,random_state=True,n_init=10)
k_means.fit(model.wv.vectors.astype("double"))

model.wv.similar_by_vector(k_means.cluster_centers_[0],topn=10,restrict_vocab=None)
o/p:('said', 0.9991175532341003),
 ('market', 0.9990715980529785),
 ('may', 0.9990324378013611),
 ('one', 0.9990300536155701),
 ('get', 0.9990085959434509),
 ('price', 0.9989702105522156),
 ('need', 0.9988904595375061),
 ('information', 0.9988885521888733),
 ('deal', 0.9988303780555725),
 ('service', 0.9988126754760742)]

model.wv.similar_by_vector(k_means.cluster_centers_[1],topn=15,restrict_vocab=None)
o/p:[('may', 0.9991060495376587),
 ('one', 0.9990613460540771),
 ('said', 0.9990149140357971),
 ('market', 0.9989492297172546),
 ('price', 0.9989445805549622),
 ('get', 0.9989140033721924),
 ('deal', 0.9989049434661865),
 ('need', 0.9987766742706299),
 ('power', 0.9987291097640991),
 ('day', 0.998721182346344),
 ('would', 0.9987199306488037),
 ('state', 0.9986669421195984),
 ('also', 0.998647153377533),
 ('week', 0.998638391494751),
 ('information', 0.9986299276351929)]
