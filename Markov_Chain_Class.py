#!/usr/bin/env python
# coding: utf-8

# ## Markov Chain
# - Probabistic Model for Text/Natural Language Generation
# - Simple and effective way of generating new text
#     - Text
#     - Lyrics
#     - Story/Novel
#     - Code

# In[ ]:


text = "the man was ....they...then.... the ... the  "

# X is the sequence of 'K = 3' and Y is predicted character or K+1 the character

X      Y     Freq
the    " "    4
the    "n"    2
the    "y"    1
the    "i"    1
man    "_"    1


# In[17]:


def generateTable(data,k=4):
    
    T = {}
    for i in range(len(data)-k):
        X = data[i:i+k]
        Y = data[i+k]
        #print("X  %s and Y %s  "%(X,Y))
        
        if T.get(X) is None:
            T[X] = {}
            T[X][Y] = 1
        else:
            if T[X].get(Y) is None:
                T[X][Y] = 1
            else:
                T[X][Y] += 1
    
    return T
        
    


# In[19]:


T = generateTable("hello hello helli")
print(T)


# In[22]:


def convertFreqIntoProb(T):     
    for kx in T.keys():
        s = float(sum(T[kx].values()))
        for k in T[kx].keys():
            T[kx][k] = T[kx][k]/s
                
    return T


# In[24]:


T = convertFreqIntoProb(T)
print(T)


# In[78]:


text_path = "english_speech_2.txt"
def load_text(filename):
    with open(filename,encoding='utf8') as f:
        return f.read().lower()
    
text = load_text(text_path)
#text = load_text("sample_code.txt")


# In[79]:


print(text[:1000])


# ## Train our Markov Chain

# In[80]:


def trainMarkovChain(text,k=4):
    
    T = generateTable(text,k)
    T = convertFreqIntoProb(T)
    
    return T
    


# In[81]:


model = trainMarkovChain(text)


# In[82]:


print(model)


# ## Generate Text at Text Time!
# 

# In[83]:


import numpy as np


# In[84]:


# sampling !
fruits = ["apple","banana","mango"]
prob = ["0.8",".1","0.1"]
for i in range(10):
    #sampling according a probability distribution
    print(np.random.choice(fruits,p=prob))


# In[85]:


def sample_next(ctx,T,k):
    ctx = ctx[-k:]
    if T.get(ctx) is None:
        return " "
    possible_Chars = list(T[ctx].keys())
    possible_values = list(T[ctx].values())
    
    #print(possible_Chars)
    #print(possible_values)
    
    return np.random.choice(possible_Chars,p=possible_values)


# In[86]:


sample_next("comm",model,4)


# In[87]:


def generateText(starting_sent,k=4,maxLen=1000):
    
    sentence = starting_sent
    ctx = starting_sent[-k:]
    
    for ix in range(maxLen):
        next_prediction = sample_next(ctx,model,k)
        sentence += next_prediction
        ctx = sentence[-k:]
    return sentence


# In[88]:



text = generateText("dear",k=4,maxLen=2000)
print(text)


# In[ ]:





# 

# ![](modi.gif)

# In[ ]:




