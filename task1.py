#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 1. Basic Text Pre-processing
# #### Student Name: XXXX XXXX
# #### Student ID: 000000
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
# <span style="color: red"> Note that this is a sample notebook only. You will need to fill in the proper markdown and code blocks. You might also want to make necessary changes to the structure to meet your own needs. Note also that any generic comments written in this notebook are to be removed and replace with your own words.</span>

# ## Importing libraries 

# In[140]:


# Code to import libraries as you need in this assessment, e.g.,
import nltk
import os
import numpy as np
import pandas as pd


# ### 1.1 Examining and loading data
# - xamine the data folder, including the categories and job advertisment txt documents, etc. Explain your findings here, e.g., number of folders and format of txt files, etc.
# - Load the data into proper data structures and get it ready for processing.
# - Extract webIndex and description into proper data structures.
# 

# In[141]:


# Code to inspect the provided data file...
import pandas as pd
from sklearn.datasets import load_files

job_category = load_files(r"review_job/jobcategory")# load data file


# In[8]:


print(job_category)


# In[14]:


job_category.keys()


# In[15]:


raw_data = job_category['data']
raw_data


# In[16]:


job_category['DESCR']


# In[103]:


# creating catergory column 
filenames=job_category['filenames']
final_filenames=[]
for i in filenames:
    var=i.split("/")
    final_filenames.append(var)


# In[142]:


cat_var=[]
for i in range(776):
    cat_var.append(final_filenames[i][2])


# In[143]:


cat_var


# In[18]:


job_category['target']


# In[144]:


# converting bytes to string 
data = []
for n in raw_data:
    data.append(n.decode('utf-8'))
data


# In[145]:


#split the lines and store in r_data
r_data = []
for line in data:
    r_data.append(line.split("\n"))

r_data


# In[146]:


# converting in to diictionary
results = []
for i in r_data:
    results.append(dict())
    for key_value in i:
        key, value =key_value.split(':', 1)  # removing surrounding whitespace
        results[-1][key] = value
results


# In[147]:


# creating a data frame
data_frame= pd.DataFrame(results)
data_frame


# In[148]:


data_frame =data_frame.reset_index(drop = True)
data_frame


# In[36]:


data_frame.Description[0:776]


# In[149]:


#converting the description to tolist
#tolist= data_frame.Description
description = data_frame.Description.tolist()
description


# In[150]:


emp = 10
description[10]


# In[151]:


emp = 10
job_category['filenames'][emp], job_category['target'][emp]


# ### 1.2 Pre-processing data
# Perform the required text pre-processing steps.

# In[152]:


from nltk import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from itertools import chain

# tokenize the description column
def token_description(raw_description):
    
    description = raw_description.lower()
    
    #sentences = sent_tokenize(description)
    
    pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
    
    tokenizer = RegexpTokenizer(pattern)
    
    token_lists = tokenizer.tokenize(description)
    
    return token_lists    


# In[153]:


tk_description = [token_description(r) for r in description]


# In[154]:


tk_description


# In[155]:


for i in range(0, len(tk_description)):
    print("\n\n",i,"\n\n")
    print(tk_description[i])


# In[44]:


for i in range(0, len(tk_description)):
    tokenised_description = list(chain.from_iterable(tk_description))
    print(tokenised_description)


# In[156]:


print("Raw :\n",description[emp],'\n')
print("Tokenized :\n",tk_description[emp])


# ## removing the words with length less than 2

# In[157]:


# removing the words with length less than 2
tk_description = [[ d for d in describe if len(d)>=2]          for describe in tk_description]
tk_description


# In[47]:


print("Tokenized :\n",tk_description[emp])


# #Remove stopwords using the provided stop words list 

# In[158]:


stopwords_en = []
with open('./stopwords_en.txt') as f:
    stopwords_en = f.read().splitlines()
print(stopwords_en)


# In[159]:


stopwords_en


# In[50]:


len(stopwords_en)


# In[51]:


# filter out stop words
filtered_words = [[s for s in token if s not in stopwords_en]        for token in tk_description] 
filtered_words


# In[160]:


len(filtered_words)


# In[161]:


# from itertools import chain
# token_describe = list(chain.from_iterable(filtered_words))
token_describe = filtered_words
#len(tokens)
token_describe


# In[163]:


from nltk.probability import *


# In[164]:


words =list(chain.from_iterable(token_describe))

#vocab = set(words)
words


# In[165]:


term_fd = FreqDist(words)
term_fd

#term_fd.most_common(25)


# In[166]:


lessFreqWords = set([k for k, v in term_fd.items() if v<2])


# In[167]:


lessFreqWords


# In[169]:


token_describe


# #Remove the word that appears only once in the document collection, based on term frequency

# In[170]:


token_description = [[w for w in review if w not in lessFreqWords]    for review in token_describe]
len(token_description)


# In[171]:


words = list(chain.from_iterable(token_description)) # we put all the tokens in the corpus in a single list
words


# In[174]:


len(token_description)
token_description


# #Remove the top 50 most frequent words based on document frequency.

# In[176]:


words_2 = list(chain.from_iterable([set(describe) for describe in token_description]))
doc_fd = FreqDist(words_2)
most_doc_fd = doc_fd.most_common(50)
most_doc_fd


# In[177]:


token_description_new = [[w for w in most if w not in most_doc_fd ]    for most in token_description]
token_description_new


# In[178]:


data_frame['category'] = cat_var


# In[179]:


data_frame['token_description'] = token_description_new
data_frame


# In[128]:


data_frame.to_csv('job.csv', index= False)


# In[181]:


token_describe = list(chain.from_iterable(token_description_new))
token_describe


# In[182]:


token_describe


# #### Save all job advertisement text and information in txt file

# In[184]:


def save_description(description_Filename, token_description_new):
    out_file = open(description_Filename, 'w')
    string = "\n".join([" ".join (describe)  for describe in  token_description_new])
    out_file.write(string)
    out_file.close()


# In[185]:


save_description('description.txt',token_description_new)


# In[186]:


# code to perform the task...


# ## Saving required outputs
# Save the vocabulary, bigrams and job advertisment txt as per spectification.
# - vocab.txt

# In[187]:


# code to save output data...
words = list(chain.from_iterable(token_description_new))
vocab = sorted(list(set(words)))
len(vocab)
vocab
vocabulary = {k: v for v, k in enumerate(vocab)}
vocabulary     


# In[188]:


def save_vocabulary(vocabularyfilename, vocabulary):
    out_file = open(vocabularyfilename, 'w')
    #for a in range(0, len(vocabulary)):
    for k, vocab_describe in enumerate(vocabulary):
        value = vocab_describe
        out_file.write("{}:{}\n".format(value, k))
    out_file.write('\n')
    out_file.close
save_vocabulary('vocab.txt', vocabulary)    


# ## Summary
# For all the words that we removed (including step 4,5,6,7), you will also exclude them in the generated
# vocabulary.

# ## Couple of notes for all code blocks in this notebook
# - please provide proper comment on your code
# - Please re-start and run all cells to make sure codes are runable and include your output in the submission.   
# 
