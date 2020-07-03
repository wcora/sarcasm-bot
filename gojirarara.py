#!/usr/bin/env python
# coding: utf-8

# In[16]:


# link to the model in case you need it for testing: 
# model is under hcii-assignment4 subfolder
# https://drive.google.com/drive/folders/1ZDG6EIVRn5nDgzOMWIdPdUB5hJGOgGX0?usp=sharing

import tweepy
import credentials
import numpy as np

auth = tweepy.OAuthHandler(credentials.consumer_key, credentials.consumer_secret)
auth.set_access_token(credentials.access_token, credentials.access_token_secret)
api = tweepy.API(auth)

user = api.me()
print (user.name)


# In[21]:


from fastai.text import *
import pandas as pd
from pathlib import Path
data_path = Path('./haii-assignment4')
serve_classifier = load_learner(path=data_path, file='satire_awd.pkl')
serve_lm = load_learner(path=data_path, file='headlines-lm.pkl')


# In[66]:



####
# Define the search
#####
query = '@bot_gojirarara'
max_tweets = 100

####
# Do the search
#####
searched_tweets = []
tweets = []

last_id = -1
while len(searched_tweets) < max_tweets:
    count = max_tweets - len(searched_tweets)
    try:
        new_tweets = api.search(q=query, count=count, max_id=str(last_id - 1))
        if not new_tweets:
            break
        searched_tweets.extend(new_tweets)
        last_id = new_tweets[-1].id
    except tweepy.TweepError as e:
        # depending on TweepError.code, one may want to retry or wait                                                                                                                 
        # to keep things simple, we will give up on an error                                                                                                                          
        break    
    
####
# Iterate over the search
#####
for status in searched_tweets:
    print	(status.text)
    tweets.append(status.text)


# In[67]:


# revised code: 
# after parsing by @bot_gojirarara, if users input a string before mentioning my bot (e.g. 'i want to blahblah @bot_gojirarara')
# my bot will store and analyze that string before @bot_gojirarara, 
# if there is no string (-> '')  before mentioning my bot (e.g. '@bot_gojirarara I want to blahblahblah'), 
# it will store and analyze the string after @bot_gojirarara

for i in range(len(tweets)):
    tw = []
    tw = tweets[i].split("@bot_gojirarara")
    if len(tw) > 1:
        if (tw[0] == ''):
            tweets[i] = (tweets[i].split("@bot_gojirarara ")[1])
        else:
            tweets[i] = (tweets[i].split("@bot_gojirarara")[0])
    
tweets


# In[68]:


str(serve_classifier.predict(tweets[7])[0])


# In[69]:


for i in range(len(tweets)):
    t = []
    t = tweets[i].split(' ')
    t2 = ''
    if (len(t) >= 2):
        t2 +=  t[0] 
        t2 += ' '
        t2 += t[1] 
    if (len(t) >= 3):
        t2 += ' '
        t2 += t[2]
        
    if str(serve_classifier.predict(tweets[i])[0]) == '0':
        api.update_status(
          'It looks real, not a satire. @' + searched_tweets[i].author.screen_name + ' This is what I want to say: '+serve_lm.predict(t2, n_words=7),
          searched_tweets[i].id_str
        ) 
    else:
          api.update_status(
          'It looks like you are being sarcastic! @' + searched_tweets[i].author.screen_name + ' This is what I want to say: '+serve_lm.predict(t2, n_words=7),
          searched_tweets[i].id_str
        )


# # Reflect: Would you recommend using our satire-classifier as a good starting point to build a fake-news classifier? (10 points)
# 
# If so, what changes would we need to make to make it useful for this purpose? If not, why not? 
# 
# 
# Add reflections to the bot notebook (10 points)

# # Reflection
# 
# In general, satirical news is often associated with ingenuousness. One of our dataset - the Onion is a source of satirical news. Based on my web search, I think it could also serve as a good source of fake news (I found lots of people referenced it in fake-news related articles and used it in building fake-news classifiers) and we could use 'Not the Onion' as the resource of genuine news (also frequently referenced by people in making fake-news classifier). 
# 
# Using an approach very similar to our satire classifier, we could label news from Not the Onion as 0 (i.e. genuine news) and news from the Onion as 1 (i.e. fake news), and train the language model to let it predict the fake/genuine label in the end, in addition to predicting the next word. 
# 
# Change to make:
# 
# In this case, I think one issue that we must take a closer look at is the false positive vs. false negative rate. Though model accuracy is important, but in case of fake news detector, the cost of false negative of this fake news classifier (i.e. predicting a headline as genuine but it is actually a fake news) is definitely much bigger than that of a satire classifier. It is okay if our bot cannot get the jokes, but it would cause big problem if it tells people a news is trustworthy while it is fake. Therefore, for fake news classifier we should try to pull the confusion matrix and make sure it has a low false negative rate.

# # Extra credit: Test with Users and Iterate (5 points)
# In this part, you’ll ask three participants to interact with your bot. You’ll give the user high-level information about what the domain of the bot is, and then see how they interact with it. Ask each of the participants to ask your chatbot at least three different things. Record how they interact with your bot. After this participant input, update your bot to attempt to address how that participant interacted with your chatbot. 
# 
# 
# Add to bot notebook: How did what your participants input compare to the ones you tested so far? How did participants react when the chatbot didn’t respond correctly, or responded with nonsense? (2.5 points)
# 
# 
# Add to bot notebook: what change could you make in response to this feedback? (2.5 points)
# 
# 

# # User testing
# 
# I told my friends that I built a chatbot to detect satires. I created three twitter accounts (unfortunately none of us use twitter), handed them the passwords, and let them find headlines and mention my chatbot. I did suggest them to from good ones from either the Onion or the Daily Mash and mention my chatbot to help them detect if it is a joke or not. Both sources have a good amount of satirical news.
# 
# ### Part 1:  How did what your participants input compare to the ones you tested so far? How did participants react when the chatbot didn’t respond correctly, or responded with nonsense? (2.5 points)
# 
# #### Compare user input with my input
# 
# * My friends tried to find some funny headlines from the two sources I gave them, but one of them also tried to say something like "hey what can you tell me about?" even I told them my chatbot is not built for answering questions and their questions will be treated as a headline. 
# 
# I think this has to do with the mental model of users. Users have no knowledge of how the bot is built and they cannot tell what's the difference between analyzing their questions as questions and analyzing their questions as a news headline. 
# 
# * My friends do not have as good knowledge as I do with the chatbot. So while I have instructed them to type "@bot_gojirarara" first and then type a headline, some of them still wrote their sentence first and then @bot_gojirarara. 
# 
# Again, users have no knowledge of how the bot is built. They think either mentioning the bot before or after the headline should not matter, but it does. In my original design, the headline must be typed after @bot. If the headline was added before tagging the bot, it wouldn't work. 
# 
# #### Users' reaction to my chatbot's nonsense reply
# 
# * Ordinary users cannot understand what the chatbot is replying ("is it telling me a joke?? or is it writing another headline? a satire? not a satire?")
# * They also found that the chatbot produces nonsense words (xxbos) and cannot finish the sentence properly. They think those are the signs that the chatbot behaves poorly. Without understanding the algorithm behind, they feel that this bot is just producing random sentences.
# * They were actually quite impressed that the bot did pretty well in detecting their satires. But they felt that they have no idea what the reply means. They only noticed that the first three words of the chatbot's reply are the same as their previous input. But the rest of the sentence make little sense to them. 
# 
# ### Part 2:  what change could you make in response to this feedback? (2.5 points)
# 
# ####  I fixed the input format issue which caused my bot to run into error. 
# One of my friends accidentally swapped the order of mentioning the bot and writing the headline, so the headline looks like this: "Magazine blahblah @bot_gojirarara", and my bot could not give it a response properly. Therefore, in order to fix this issue, I let the chatbot check if the string before @bot_gojirarara is empty string '',  if so, my bot will store the string after @bot_gojirarara as usual, if the string before mentioning my bot is not empty string, I will store and use that non-empty string before @bot_gojirarara. (code is now modified, you can find it under comment ### revised code)
# 
# #### Another issue I realized is the difference between users' mental model and how the machine works in reality. 
# It reminds me of the google ads—users are not sure why they see certain ads and feel annoyed if they see something that they aren't actually interested. I think in terms of our chatbot, it should: 
# * Clearly tell users what it can do versus what it cannot do (such as "it is a satire detecter, you can try telling it a joke or let it analyze a satirical headline, but unfortunately, it is not able to answer your question such as 'can you tell me a joke?'")
# * Clearly define the format of which users should interact with it (you should start your sentence bt '@bot_gojirarara '") and make sure users get this information to users before they can start the conversation with the bot (for the types of chatbot that lives in a little dialog box by the corner of a webpage, the format of input should be indicated on the welcome page, and users have to select 'yes, I read through it' to continue). 
# * We could try to add a feature such as if users think the bot behaves strange/ want to know more about how the bot works, type '@bot_gojirarara What does your reply mean?' and explain that the bot used the first three words of the user input and tried to generate a headline. 

# # Extra credit: Deploy bot (10 points)
# Take your Jupyter notebook (the bot notebook) and deploy it such that it runs once an hour, and responds to all messages sent to it. 
# 
# Add to bot notebook: Twitter handle (of bot) to test for. (10 points)

# In[ ]:


#### run the following commands in terminal

# crontab -e

#### (will enter vim, press i to) add following lines 

# PATH=/Users/Melia/anaconda3/bin/
# 41 * * * * jupyter nbconvert --execute /Users/Melia/HAI/HW4/gojirarara.ipynb

#### the system will ask for permission
#### this will let the system run the notebook every hour at 41 min (1:41 then 2:41 then 3:41...)
#### testing this will need to change 1) document path and 2) executable path based on $ which jupyter


#### The following is /var/mail log to prove it worked

# From Melia@Coramelia.local  Sun Nov 10 21:41:17 2019
# Return-Path: <Melia@Coramelia.local>
# X-Original-To: Melia
# Delivered-To: Melia@Coramelia.local
# Received: by Coramelia.local (Postfix, from userid 501)
#         id 12AA9201C5CEBD; Sun, 10 Nov 2019 21:41:17 -0500 (EST)
# From: Melia@Coramelia.local (Cron Daemon)
# To: Melia@Coramelia.local
# Subject: Cron <Melia@Coramelia> jupyter nbconvert --execute /Users/Melia/HAI/HW4/gojirarara.ipynb
# X-Cron-Env: <PATH=/Users/Melia/anaconda3/bin/>
# X-Cron-Env: <SHELL=/bin/sh>
# X-Cron-Env: <LOGNAME=Melia>
# X-Cron-Env: <USER=Melia>
# X-Cron-Env: <HOME=/Users/Melia>
# Message-Id: <20191111024117.12AA9201C5CEBD@Coramelia.local>
# Date: Sun, 10 Nov 2019 21:41:01 -0500 (EST)

# [NbConvertApp] Converting notebook /Users/Melia/HAI/HW4/gojirarara.ipynb to html
# [NbConvertApp] Executing notebook with kernel: python3
# [NbConvertApp] Writing 303396 bytes to /Users/Melia/HAI/HW4/gojirarara.html

