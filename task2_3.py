#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 2&3
# #### Student Name: Harshith Hullkere Siddegowda
# #### Student ID: S3914263
# 
# Date: XXXX
# 
# Version: 1.0
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used: please include all the libraries you used in your assignment, e.g.,:
# * pandas
# * re
# * numpy
# 
# ## Introduction
# You should give a brief information of this assessment task here.
# 
# Task 2:Generating Feature Representations for Job Advertisement Descriptions
# 
# Task 3: Job Advertisement Classification 
# 

# ## Importing libraries 

# In[741]:


# importing the required libraries
import nltk
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# In[742]:


df = pd.read_csv('job.csv')
df


# In[744]:


title = df['Title']
title


# In[765]:


df.token_description


# In[766]:


# Code to import libraries as you need in this assessment, e.g.,
#reading the description text file
description = './description.txt'
with open(description) as f:
    tk_description = f.read().splitlines()
   


# In[767]:


print(len(tk_description))
emp = 10

tk_description[emp]


# ## Task 2. Generating Feature Representations for Job Advertisement Descriptions

# ...... Sections and code blocks on buidling different document feature represetations
# 

# In[768]:


# Code to perform the task...
tk_description = [describe.split() for describe in tk_description]
tk_description[emp]
tk_description


# In[783]:


#the generated Count vector representation must be based on the generated vocabulary in Task 1 (
def gen_vocIndex(voc_fname):
    with open(voc_fname) as vocf:
        vocablines = vocf.read().splitlines()
    voc_Ind = [l.split(':') for l in vocablines] 
    voc_Ind = {int(element[1]):element[0] for element in voc_Ind} 
    return voc_Ind #("{}:{}".format(voc_Ind)) 

voc_fname = './vocab.txt'
voc_dict = gen_vocIndex(voc_fname)
voc_dict


# In[775]:


from itertools import chain
words = list(chain.from_iterable([set(describe) for describe in tk_description]))

vocab = sorted(set(words))
vocab


# In[506]:


len(vocab)


# In[776]:


len(words)


# In[779]:


from sklearn.feature_extraction.text import CountVectorizer
cVectorizer = CountVectorizer(analyzer = "word", vocabulary = vocab)
count_features = cVectorizer.fit_transform([' '.join(describe) for describe in tk_description])
print(count_features.shape)


# In[781]:


print(count_features)


# In[860]:


count_vectors_webindex = df['Webindex'].astype(str)


# ## Saving Output Count_Vector.txt

# In[870]:


def write_vectorFile(data_features,filename):
    num = data_features.shape[0] # the number of document
    out_file = open(filename, 'w') # creates a txt file and open to save the vector representation
    for a_ind in range(0, num): # loop through each article by index
        out_file.write("{},".format("#"+count_vectors_webindex[a_ind]))
        for f_ind in data_features[a_ind].nonzero()[1]: # for each word index that has non-zero entry in the data_feature
            value = data_features[a_ind][0,f_ind] # retrieve the value of the entry from data_features
            out_file.write("{}:{},".format(f_ind,value)) # write the entry to the file in the format of word_index:value
        out_file.write('\n') # start a new line after each article
    out_file.close() # close the file       


# In[871]:


cVector_file ="./count_vector.txt" 
write_vectorFile(count_features, cVector_file)


# ## 2. Our in-house trained Word2vec model

# In[785]:


get_ipython().system('pip install --upgrade gensim ')
import gensim.downloader as api


# In[552]:


pip install gensim


# In[553]:


get_ipython().system('pip install python-Levenshtein')


# In[554]:


from gensim import utils


# In[555]:


class MyCorpus:
    """An iterator that yields sentences (lists of str)."""
    def __init__(self, corpusFile):
       MyCorpus.fpath = corpusFile
    def __iter__(self):
        for line in open(self.fpath):
          yield utils.simple_preprocess(line)  


# In[556]:


import gensim.models

sentences = MyCorpus('./description.txt')
description_w2v = gensim.models.Word2Vec(sentences = sentences, vector_size =100)


# In[557]:


print(description_w2v)


# In[558]:


description_W2v_wv = description_w2v.wv
print(description_W2v_wv)


# In[559]:


len(description_W2v_wv)


# In[560]:


for index, word in enumerate(description_W2v_wv.index_to_key):
    if index == 10:
        break
    print(f"word #{index}/{len(description_W2v_wv.index_to_key)} is {word}")


# In[561]:


try:
    vec = description_W2v_wv['h']
    print(vec)
except KeyError:
    print("The word 'happy' does not appear in model")


# In[786]:


# save the model
description_w2v.save("description_W2v.model")


# ## TF-IDF weighted vector

# In[790]:


from sklearn.feature_extraction.text import TfidfVectorizer
tVectorizer = TfidfVectorizer(analyzer = "word",vocabulary = vocab) # initialised the TfidfVectorizer
tfidf_features = tVectorizer.fit_transform([' '.join(describe) for describe in tk_description]) # generate the tfidf vector representation for all articles
tfidf_features.shape


# In[799]:


def write_vectorFile(data_features,filename):
    num = data_features.shape[0] # the number of document
    out_file = open(filename, 'w') # creates a txt file and open to save the vector representation
    for a_ind in range(0, num): # loop through each article by index
        for f_ind in data_features[a_ind].nonzero()[1]: # for each word index that has non-zero entry in the data_feature
            value = data_features[a_ind][0,f_ind] # retrieve the value of the entry from data_features
            out_file.write("{}:{} ".format(f_ind,value)) # write the entry to the file in the format of word_index:value
        out_file.write('\n') # start a new line after each article
    out_file.close() # close the file


# In[800]:


tVector_file = "./description_tVector.txt" # file name of the tfidf vector
write_vectorFile(tfidf_features,tVector_file) # write the tfidf vector to file


# In[801]:


def gen_vocIndex(voc_fname):
    with open(voc_fname) as vocf:
        vocablines = vocf.read().splitlines()
    voc_Ind = [l.split(':') for l in vocablines] 
    voc_Ind = {int(element[1]):element[0] for element in voc_Ind} 
    return voc_Ind #("{}:{}".format(voc_Ind)) 

voc_fname = './vocab.txt'
voc_dict = gen_vocIndex(voc_fname)


# In[802]:


voc_dict


# In[810]:


def doc_wordweights(fName_tVectors, voc_dict):
    tfidf_weights = [] # a list to store the  word:weight dictionaries of documents
    
    with open(fName_tVectors) as tVecf: 
        tVectors = tVecf.read().splitlines() # each line is a tfidf vector representation of a document in string format 'word_index:weight word_index:weight .......'
    for tv in tVectors: # for each tfidf document vector
        tv = tv.strip()
        weights = tv.split(' ') # list of 'word_index:weight' entries
        weights = [w.split(':') for w in weights] # change the format of weight to a list of '[word_index,weight]' entries
        wordweight_dict = {voc_dict[int(w[0])]:w[1] for w in weights} # construct the weight dictionary, where each entry is 'word:weight'
        tfidf_weights.append(wordweight_dict) 
    return tfidf_weights

fName_tVectors = 'description_tvector.txt'
tfidf_weights = doc_wordweights(fName_tVectors, voc_dict)


# In[811]:


tfidf_weights[0]


# In[812]:


def gen_docVecs(wv,tk_decriptions,tfidf = []):
    docs_vectors = pd.DataFrame() # creating empty final dataframe
    #stopwords = nltk.corpus.stopwords.words('english') # removing stop words

    for i in range(0,len(tk_decriptions)):
        tokens = list(set(tk_decriptions[i])) # get the list of distinct words of the document

        temp = pd.DataFrame()  # creating a temporary dataframe(store value for 1st doc & for 2nd doc remove the details of 1st & proced through 2nd and so on..)
        for w_ind in range(0, len(tokens)): # looping through each word of a single document and spliting through space
            try:
                word = tokens[w_ind]
                word_vec = wv[word] # if word is present in embeddings(goole provides weights associate with words(300)) then proceed
                
                if tfidf != []:
                    word_weight = float(tfidf[i][word])
                else:
                    word_weight = 1
                temp = temp.append(pd.Series(word_vec*word_weight), ignore_index = True) # if word is present then append it to temporary dataframe
            except:
                pass
        doc_vector = temp.sum() # take the sum of each column(w0, w1, w2,........w300)
        docs_vectors = docs_vectors.append(doc_vector, ignore_index = True) # append each document value to the final dataframe
    return docs_vectors


# In[813]:


weighted_descriptionW2v_dvs = gen_docVecs(description_W2v_wv,tk_description,tfidf_weights)


# ## TF-IDF unweighted vector

# In[814]:


def gen_docVecs(wv,tk_txts): # generate vector representation for documents
    docs_vectors = pd.DataFrame() # creating empty final dataframe
    #stopwords = nltk.corpus.stopwords.words('english') # if we haven't pre-processed the articles, it's a good idea to remove stop words

    for i in range(0,len(tk_txts)):
        tokens = tk_txts[i]
        temp = pd.DataFrame()  # creating a temporary dataframe(store value for 1st doc & for 2nd doc remove the details of 1st & proced through 2nd and so on..)
        for w_ind in range(0, len(tokens)): # looping through each word of a single document and spliting through space
            try:
                word = tokens[w_ind]
                word_vec = wv[word] # if word is present in embeddings(goole provides weights associate with words(300)) then proceed
                temp = temp.append(pd.Series(word_vec), ignore_index = True) # if word is present then append it to temporary dataframe
            except:
                pass
        doc_vector = temp.sum() # take the sum of each column
        docs_vectors = docs_vectors.append(doc_vector, ignore_index = True) # append each document value to the final dataframe
    return docs_vectors


# In[815]:


descriptionW2v_dvs = gen_docVecs(description_W2v_wv,tk_description)
descriptionW2v_dvs.isna().any().sum()
descriptionW2v_dvs


# ## task 3:
# 

# ## before applying machine learning model I am selecting 2nd language FastText model
# building second language model to perfom evaluation on the two different model performance

# In[816]:


get_ipython().system('wget http://nlp.stanford.edu/data/glove.6B.zip')
get_ipython().system('unzip -q glove.6B.zip -d glove')


# In[818]:


from gensim.models.fasttext import FastText

# 1. Set the corpus file names/path
corpus_file = './description.txt'

# 2. Initialise the Fast Text model
descriptionFT = FastText(vector_size=100) 

# 3. build the vocabulary
descriptionFT.build_vocab(corpus_file=corpus_file)

# 4. train the model
descriptionFT.train(
    corpus_file=corpus_file, epochs=descriptionFT.epochs,
    total_examples=descriptionFT.corpus_count, total_words=descriptionFT.corpus_total_words,
)

print(bbcFT)


# In[819]:


descriptionFT_wv = descriptionFT.wv


print(descriptionFT_wv)


# In[820]:


descriptionFT_wv['experience']


# In[821]:


print('accountant' in descriptionFT_wv.key_to_index)


# In[822]:


print(descriptionFT_wv.most_similar('accountant', topn = 10))


# In[823]:


descriptionFT.save('descriptionFT.model')


# In[824]:


voc_dict


# ## 2nd language choosen Fast text TF-IDF Weighted vector

# In[827]:


def doc_wordweights(fName_tVectors, voc_dict):
    tfidf_weights = [] # a list to store the  word:weight dictionaries of documents
    
    with open(fName_tVectors) as tVecf: 
        tVectors = tVecf.read().splitlines() # each line is a tfidf vector representation of a document in string format 'word_index:weight word_index:weight .......'
    for tv in tVectors: # for each tfidf document vector
        tv = tv.strip()
        weights = tv.split(' ') # list of 'word_index:weight' entries
        weights = [w.split(':') for w in weights] # change the format of weight to a list of '[word_index,weight]' entries
        wordweight_dict = {voc_dict[int(w[0])]:w[1] for w in weights} # construct the weight dictionary, where each entry is 'word:weight'
        tfidf_weights.append(wordweight_dict) 
    return tfidf_weights

fName_tVectors = 'description_tvector.txt'
tfidf_weights = doc_wordweights(fName_tVectors, voc_dict)


# In[828]:


tfidf_weights[0]


# In[829]:


def gen_docVecs(wv,tk_txts,tfidf = []): # generate vector representation for documents
    docs_vectors = pd.DataFrame() # creating empty final dataframe
    #stopwords = nltk.corpus.stopwords.words('english') # removing stop words

    for i in range(0,len(tk_txts)):
        tokens = list(set(tk_txts[i])) # get the list of distinct words of the document

        temp = pd.DataFrame()  # creating a temporary dataframe(store value for 1st doc & for 2nd doc remove the details of 1st & proced through 2nd and so on..)
        for w_ind in range(0, len(tokens)): # looping through each word of a single document and spliting through space
            try:
                word = tokens[w_ind]
                word_vec = wv[word] # if word is present in embeddings(goole provides weights associate with words(300)) then proceed
                
                if tfidf != []:
                    word_weight = float(tfidf[i][word])
                else:
                    word_weight = 1
                temp = temp.append(pd.Series(word_vec*word_weight), ignore_index = True) # if word is present then append it to temporary dataframe
            except:
                pass
        doc_vector = temp.sum() # take the sum of each column(w0, w1, w2,........w300)
        docs_vectors = docs_vectors.append(doc_vector, ignore_index = True) # append each document value to the final dataframe
    return docs_vectors


# In[830]:


weighted_descriptionFT_dvs = gen_docVecs(descriptionFT_wv,tk_description,tfidf_weights)
weighted_descriptionFT_dvs


# In[637]:


def gen_docVecs(wv,tk_txts): # generate vector representation for documents
    docs_vectors = pd.DataFrame() # creating empty final dataframe
    #stopwords = nltk.corpus.stopwords.words('english') # if we haven't pre-processed the articles, it's a good idea to remove stop words

    for i in range(0,len(tk_txts)):
        tokens = tk_txts[i]
        temp = pd.DataFrame()  # creating a temporary dataframe(store value for 1st doc & for 2nd doc remove the details of 1st & proced through 2nd and so on..)
        for w_ind in range(0, len(tokens)): # looping through each word of a single document and spliting through space
            try:
                word = tokens[w_ind]
                word_vec = wv[word] # if word is present in embeddings(goole provides weights associate with words(300)) then proceed
                temp = temp.append(pd.Series(word_vec), ignore_index = True) # if word is present then append it to temporary dataframe
            except:
                pass
        doc_vector = temp.sum() # take the sum of each column
        docs_vectors = docs_vectors.append(doc_vector, ignore_index = True) # append each document value to the final dataframe
    return docs_vectors


# In[638]:


descriptionFT_wv_dvs = gen_docVecs(descriptionFT_wv,tk_description)
descriptionFT_wv_dvs.isna().any().sum() # check whether there is any null values in the document vectors dataframe.


# ## Task 3. machine learning models Job Advertisement Classification

# In[831]:


df['Title']
len(df['Title'])


# #### machine learning models  (TF IDF weighted)based on the feature representations of the documents

# In[832]:


get_ipython().run_line_magic('matplotlib', 'inline')
seed= 0
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

models = [weighted_descriptionW2v_dvs, weighted_descriptionFT_dvs]
models_names = ["weighted_descriptionW2v_dvs","weighted_descriptionFT_dvss"]
for i in range(0, len(models)):
    dv = models[i]
    name = models_names[i]
    features = dv.to_numpy()
    print(name + " tSNE 2dimensional projected Feature space")
    
    plotTSNE(df['category'],features)
    
    X_train, X_test, Y_train, Y_test, train_indices,test_indices = train_test_split(dv, df['category'], list(range(0, len(df))), test_size = 0.10, random_state = 77)
    
    model = LogisticRegression(max_iter =2000, random_state = seed)
    model.fit(X_train, Y_train)
    print("Accuracy:", model.score(X_test, Y_test))
    print('\n\n')
    


# In[833]:


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
def plotTSNE(labels,features): # features as a numpy array, each element of the array is the document embedding of an article
    categories = sorted(labels.unique())
    # Sampling a subset of our dataset because t-SNE is computationally expensive
    SAMPLE_SIZE = int(len(features) * 0.3)
    np.random.seed(0)
    indices = np.random.choice(range(len(features)), size=SAMPLE_SIZE, replace=False)
    projected_features = TSNE(n_components=2, random_state=0).fit_transform(features[indices])
    colors = ['pink', 'green', 'midnightblue', 'orange', 'darkgrey']
    for i in range(0,len(categories)):
        points = projected_features[(labels[indices] == categories[i])]
        plt.scatter(points[:, 0], points[:, 1], s=30, c=colors[i], label=categories[i])
    plt.title("Feature vector for each article, projected on 2 dimensions.",
              fontdict=dict(fontsize=15))
    plt.legend()
    plt.show()


# #### machine learning models  (unweighted)

# In[834]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# creating training and test split
X_train, X_test, y_train, y_test,train_indices,test_indices = train_test_split(descriptionW2v_dvs, df['category'], list(range(0,len(df))),test_size=0.33, random_state=seed)

model = LogisticRegression(max_iter = 2000,random_state=seed) # increase the max_iter to 2000 for convergence
model.fit(X_train, y_train)
model.score(X_test, y_test)


# In[835]:


features = descriptionW2v_dvs.to_numpy() # convert the document vector dataframe to a numpy array
plotTSNE(df['category'],features) # plot the tSNE to have a look


# In[836]:


# creating training and test split
X_train, X_test, y_train, y_test,train_indices,test_indices = train_test_split(descriptionFT_wv_dvs, df['category'], list(range(0,len(df))),test_size=0.33, random_state=seed)

model = LogisticRegression(max_iter = 2000,random_state=seed)
model.fit(X_train, y_train)
model.score(X_test, y_test)    


# In[837]:


features = descriptionFT_wv_dvs.to_numpy() # convert the document vector dataframe to a numpy array
plotTSNE(df['category'],features)


# ## cross validation

# In[838]:


from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)


# In[839]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

categories = ['Neg','Pos'] # this gives sorted set of unique label names

sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=categories, yticklabels=categories) # creates a heatmap from the confusion matrix
plt.ylabel('Actual')
plt.xlabel('Predicted')


# ### 5-Fold Cross Validation

# In[840]:


from sklearn.model_selection import KFold
num_folds = 5
kf = KFold(n_splits= num_folds, random_state=seed, shuffle = True) # initialise a 5 fold validation
print(kf)


# In[841]:


def evaluate(X_train,X_test,y_train, y_test,seed):
    model = LogisticRegression(random_state=seed,max_iter = 1000)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


# In[842]:


import pandas as pd
from sklearn.linear_model import LogisticRegression

num_models = 3
cv_df = pd.DataFrame(columns = ['count','tfidf'],index=range(num_folds)) # creates a dataframe to store the accuracy scores in all the folds

fold = 0
for train_index, test_index in kf.split(list(range(0,len(tk_description)))):
    y_train = [str(tk_description[i]) for i in train_index]
    y_test = [str(tk_description[i]) for i in test_index]
    
    X_train_count, X_test_count = count_features[train_index], count_features[test_index]
    cv_df.loc[fold,'count'] = evaluate(count_features[train_index],count_features[test_index],y_train,y_test,seed)

    X_train_tfidf, X_test_tfidf = tfidf_features[train_index], tfidf_features[test_index]
    cv_df.loc[fold,'tfidf'] = evaluate(tfidf_features[train_index],tfidf_features[test_index],y_train,y_test,seed)
    
    fold +=1


# ## Information Provide Higher Accuracy

# In[650]:


title = df.Title.tolist()
title


# In[653]:


from nltk import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from itertools import chain


def token_title(raw_description):
    
    description = raw_description.lower()
    #sentences = sent_tokenize(description)
    pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
    tokenizer = RegexpTokenizer(pattern)
    #token_lists = [tokenizer.tokenize(sen) for sen in description]
    token_lists = tokenizer.tokenize(description)
    
    #tokenised_description = list(chain.from_iterable(token_lists))
    return token_lists


# In[657]:


tk_title = [token_title(r) for r in  title]
tk_title


# In[658]:


tk_title = [[ t for t in title if len(t)>=2]          for title in tk_title]
tk_title


# In[660]:


stopwords_en = []
with open('./stopwords_en.txt') as f:
    stopwords_en = f.read().splitlines()
print(stopwords_en)


# In[676]:


filtered_title = [[s for s in token if s not in stopwords_en]        for token in tk_title ] 
filtered_title
token_title = filtered_title
token_title


# In[671]:


from nltk.probability import *
words_title =list(chain.from_iterable(token_title))

#vocab = set(words)
words_title


# In[672]:


term_fd = FreqDist(words_title)
term_fd


# In[673]:


lessFreqWords = set([k for k, v in term_fd.items() if v<2])
lessFreqWords


# In[677]:


token_title = [[w for w in title if w not in lessFreqWords]    for title in token_title]
len(token_title)
token_title


# In[678]:


words_2 = list(chain.from_iterable([set(title) for title in token_title]))
doc_fd = FreqDist(words_2)
most_doc_fd = doc_fd.most_common(50)
most_doc_fd


# In[986]:


token_title_new = [[w for w in most if w not in most_doc_fd ]    for most in token_title]
token_title_new


# In[987]:


for x in token_title_new:
    
    if len(x)==0:
        
        x.append('job')

token_title_new        


# In[988]:


res = list(filter(None, token_title_new))
token_title_n = res


# In[989]:


def save_description(description_Filename, token_description_n):
    out_file = open(description_Filename, 'w')
    string = "\n".join([" ".join (title)  for title in  token_title_n])
    out_file.write(string)
    out_file.close()


# In[990]:


save_description('tk_title.txt',token_title_n)


# In[991]:


words = list(chain.from_iterable(token_title_new))
vocab = sorted(list(set(words)))

len(vocab)
vocab


vocabulary = {k: v for v, k in enumerate(vocab)}
vocabulary


# In[992]:


def save_vocabulary(vocabularyfilename, vocabulary):
    out_file = open(vocabularyfilename, 'w')
    #for a in range(0, len(vocabulary)):
    for k, vocab_describe in enumerate(vocabulary):
        value = vocab_describe
        out_file.write("{}:{}\n".format(value, k))
    out_file.write('\n')
    out_file.close
save_vocabulary('vocabulary.txt', vocabulary)    


# In[993]:


df['tk_title']= token_title_new


# In[994]:


df


# In[995]:


from gensim import utils
class MyCorpus:
    """An iterator that yields sentences (lists of str)."""
    def __init__(self, corpusFile):
       MyCorpus.fpath = corpusFile
    def __iter__(self):
        for line in open(self.fpath):
          yield utils.simple_preprocess(line)  


# In[996]:


import gensim.models

sentences = MyCorpus('tk_title.txt')
title_w2v = gensim.models.Word2Vec(sentences = sentences, vector_size =100)


# In[997]:


print(title_w2v)


# In[998]:


title_W2v_wv = title_w2v.wv


# In[999]:


len(title_W2v_wv)


# In[1000]:


title_w2v.save("title_W2v.model")


# In[1001]:


def gen_docVecs(wv,tk_txts): # generate vector representation for documents
    docs_vectors = pd.DataFrame() # creating empty final dataframe
    #stopwords = nltk.corpus.stopwords.words('english') # if we haven't pre-processed the articles, it's a good idea to remove stop words

    for i in range(0,len(tk_txts)):
        tokens = tk_txts[i]
        temp = pd.DataFrame()  # creating a temporary dataframe(store value for 1st doc & for 2nd doc remove the details of 1st & proced through 2nd and so on..)
        for w_ind in range(0, len(tokens)): # looping through each word of a single document and spliting through space
            try:
                word = tokens[w_ind]
                word_vec = wv[word] # if word is present in embeddings(goole provides weights associate with words(300)) then proceed
                temp = temp.append(pd.Series(word_vec), ignore_index = True) # if word is present then append it to temporary dataframe
            except:
                pass
        doc_vector = temp.sum() # take the sum of each column
        docs_vectors = docs_vectors.append(doc_vector, ignore_index = True) # append each document value to the final dataframe
    return docs_vectors


# In[1008]:


# NOTE this can take some time to finish running
titleW2v_dvs = gen_docVecs(title_W2v_wv,token_title_new)
titleW2v_dvs
#preTW2v_dvs.isna().any().sum() # check whether there is any null values in the document vectors dataframe.


# In[1007]:


features = titleW2v_dvs.to_numpy() # convert the document vector dataframe to a numpy array


# In[1009]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

categories = ['Neg','Pos'] # this gives sorted set of unique label names

sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=categories, yticklabels=categories) # creates a heatmap from the confusion matrix
plt.ylabel('Actual')
plt.xlabel('Predicted')

