#!/usr/bin/env python
# coding: utf-8

# In[4]:


from fastai.text import *
import pandas as pd

# link to the model in case you need it for testing: 
# model is under hcii-assignment4 subfolder
# https://drive.google.com/drive/folders/1ZDG6EIVRn5nDgzOMWIdPdUB5hJGOgGX0?usp=sharing


# # Language models
# dfdsf
# 
# Maybe here: https://raw.githubusercontent.com/mkearney/trumptweets/master/data/trumptweets-1515775693.tweets.csv
# 
# A language model is an algorithm that takes a sequence of words, and outputs the likely next word in the sequence. Most language models output a list of words, each with its probability of occurance. For example, if we had a sentence that started `I would like to eat a hot`, then ideally the algorithm would predict that  the word `dog` had a much higher chance of being the next word than the word `meeting`. 
# 
# Language models are a very powerful building block in natural language processing. They are used for classifying text (e.g. is this review positive or negative?), for answering questions based on text (e.g. "what is the capital of Finland?" based on the Wikipedia page on Finland), and language translation (e.g. English to Japanese).
# 
# ## The intuition behind why language models are so broadly useful
# How can this simple sounding algorithm be that broadly useful? Intuitively, this is because predicting the next word in a sentence requires a lot of information, not just about grammar and syntax, but also about semantics: what things mean in the real-world. For instance, we know that `I would like to eat a hot dog` is semantically reasonable, but `I would like to eat a hot cat` is nonsensical. 
# 
# I trained a simple language model, and asked it to predict the word following `I would like to eat a `. 
# 
# We get:
#     

# # Step 1: Load all the data 
# In this example, we are going to use a dataset of tweets from [the Onion](https://www.theonion.com), as well as some non-sarcastic news sources. I found this data set on [Kaggle](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection). 
# 
# Before I started creating this notebook, I downloaded the JSON file to a folder `haii-assignment4'
# 

# In[5]:


from pathlib import Path
data_path = Path('./haii-assignment4')


# The data is in a JSON file, so I am using the `read_json` method. If your data is CSV, use the `read_csv` method instead. 
# 
# We use the `lines=True` argument here because the author formatted each line as a separate JSON object. I think at least half of your time as a data scientist/AI researcher is spent dealing with other people's data formats!
# 

# In[6]:


df = pd.read_json(data_path/'Sarcasm_Headlines_Dataset_v2.json', lines=True)


# In[7]:


df


# As you can see, some of this dataset is drawn from the onion, the rest is drawn from places like the Huffington Post which publish real news, not satire. 

# ## Step 1a: Examine the data set (5 points)
# 
# Before we go off adventuring, let's first see what this dataset looks like. 

# ### Q: How large is this dataset? Is it balanced? (1 points)

# In[62]:


# Insert code here to check size of dataset, and how many are positive (is_sarcastic = 1) and how many negative?
# Hint: Your output will look like this.

df.is_sarcastic.value_counts()


# # Answer: 
# There are 14985 negatives and 13634 positives, it is pretty much balanced.

# ### Q: How long on average is each headline? (4 points)
# Longer text = more information. We want to see what the length of the headline is in order to see how much information it may have. 

# In[61]:


# Insert code here to find the average length of headline (in words)
## Hint: see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.count.html 
# the '\s' regex looks for spaces.

df.headline.str.count('\s').describe()


# # Answer: 
# Each headline has around 9.05 spaces, which means, on average, each headline has around 10 words. 

# # Step 2: Build a language model that knows how to write news headlines
# 
# This is the first step of our project that will be using a machine learning model. 
# 
# We are going to use the [fast.ai](https://fast.ai/) library to create this model. If you need help with understanding this section, look at the fast.ai documentation -- it is fantastic! The steps below are modified from the [online tutorial](https://docs.fast.ai/text.html#Quick-Start:-Training-an-IMDb-sentiment-model-with-ULMFiT)

# In[8]:


import fastai
from fastai.text import * 


# *Note: if this import fails for you, make sure you've installed fastai first. Do that by creating a new cell, and typing `!pip install fastai`*

# In[9]:


data_lm = (TextList.from_df(df, path=data_path, cols='headline').split_none().label_for_lm().databunch())


# ## So here is what happened above. 
# 
# First, we tell fastai that we want to work on a list of texts (headlines in our case), that are stored in a dataframe (that's the `TextList.from_df` part.) We also pass in our data path, so after we process our data, we can store it at that location. Finally, we tell it where to look for the headline in the dataframe (which column to use, `cols=`). 
# 
# Then there are two other important parts. We'll take it from the end. A `databunch` is a fastai convenience. It keeps all your training, validation and test data together. But what kind of validation data do we need for a language model? Remember that a language model predicts the next word in an input sequence of words. So, we can't just take some of the headlines and set them aside as validation. Instead, we want to use all the sentences and validate whether we can guess the right next word some fraction of the time. So, we first say `split_none` so you use all your data. Then we say `label_for_lm` so it labels the "next word" as the label for each sequence of words. It's a clever method -- see the source if you're curious!
# 

# In[10]:


data_lm.save('data_lm_export.pkl')


# Let's save this databunch. We'll use this saved copy later. 

# ## Step 2a: Learn the model
# 
# Now that we have the data, it's time to train the model.
# 
# Now, we *could* learn a language model from scratch. But we're instead going to cheat. We're going to use a pretrained language model, and finetune it for our purpose. Specifically, we're going to use a model trained on the `Wikitext-103` corpus. 
# 
# One way to understand it is to think of our pre-trained model is as a model that can predict the next word in a Wikipedia article. We want to train it to write headlines instead. Since headlines still have to sound like English, ie. follow grammar, syntax, be generally plausible etc, being able to predict the next word in Wikipedia is super useful. It allows us to start with a model that already knows some English, and then just train it for writing headlines.
# 
# 

# In[11]:


learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)


# This `AWD_LSTM` is the pretrained Wikipedia model.

# Let's train it.

# In[12]:


learn.fit_one_cycle(1, 1e-2)


# Once trained, it's time to write some headlines! We give it a starting sequence `Students protest ` and see what it comes up with. 

# In[13]:


learn.predict("Students protest ", n_words=5, no_unk=True)


# Pretty good, huh? 

# In[14]:


learn.predict('The Fed is expected to', n_words=3, no_unk=True)


# OK, it's not perfect! Let's make it a little better. 
# 
# The `unfreeze` below is telling fastai to allow us to change the weights throughout the model. We do this when we want to make the model generate text that's more similar to our headlines (than to Wikipedia). 

# In[15]:


learn.unfreeze()


# In[16]:


learn.fit_one_cycle(cyc_len=1, max_lr=1e-3)


# In[17]:


learn.predict('New Study', n_words=5)


# In[18]:


learn.predict('16 Problems', n_words=5)


# OK, now let's save our hard work. We'll use this later. (Pssst: why is it called an encoder? Look at the Fastai docs to find out!)

# In[19]:


learn.save_encoder('headlines-awd.pkl')


# Note that we also want to save the whole model, so we can reuse it in our twitter bot. 
# 

# In[20]:


learn.export('headlines-lm.pkl')


# # Step 2b: See how well the language model works (15 points)
# 
# Try generating a few more headlines. Then, answer the following questions. Wherever possible, show what code you ran, or what predictions you asked it for. *Suggestion: Try using punctuations, numbers, texts of different lengths etc.*

# ### Q: What is the effect of starting with longer strings? (5 points)
# 
# We could start our headline generation with just one word, e.g. `learn.predict('White', n_words=9)` or with many: `learn.predict('White House Says Whistleblower Did', n_words=5)`. What is the difference you see in the kinds of headlines generated?

# # Answer:
# I feel like feeding it with shorter strings give it more variabilities in the rest of the sentence. It could by chance write a good sentence (e.g. 'White house is suffering from too many needed medical measures'), but it could also write something that makes no sense at all (e.g. "White poll n shows americans ' maturity as one and"). I think the reason is that the language model predicts every word based on the previous information it has. At the beginning it does not have much information in hand, and could by chance make bad choices for the next word (e.g. "white behold../ white leg.../ white middle losers...") but if it did make a good choice of the next word (e.g. "white house...") the sentence usually tends to make good sense. 
# 
# Feeding the long strings often has good connection between adjacent words (e.g. it often predicts "not" after "did"). Since we only let it produce 5 words, there isn't much it can do, and thus produce less variability. The only problem of giving long string is the language model does not know it should finish the sentence in 5 words, and it often does not finish the sentence. 

# In[52]:


## Your answer here. Insert more cells if you want to insert code etc.
learn.predict('White', n_words=9)
#learn.predict('White House Says Whistleblower Did', n_words=5)


# In[60]:


learn.predict('White House says whistleblower did', n_words=5)


# ## Q: What aspects of the task of generating headlines does our language model do well? (5 points)
# For example, does it get grammar right? Does it know genders of people or objects? etc.

# # Your answer here. Insert more cells if you want to insert code etc.
# 
# 1. The language model is doing pretty good in matching tense. For example, when I only give it "he is" as an input, it successfully determines that a verb that comes after "he is" should be in present continuous tense. So the model outputs "he is making the first trip in australia".
# 
# 2. The language model is doing pretty good in word connections. e.g. The long string 'White House says whistleblower did' has "did" as the last word, the language model often gives "n't" as the next word to form "didn't". 
# 
# 3. The model is also capable of detecting quantity. When I feed it "we", the verb is always plural -> "we have to...", "we celebrate..." and when I feed it someone's name, it will use singular verb like "believes", "says"; when i feed it with a number like "20", it outputs a plural noun, when I feed in "one", it outputs someting like "one reporter steps..."
# 
# 4. the model can also detect gender well. When I tried people like "Trump said", the language model will correctly output "he/his" as the next word. When I tried "Taylor Swift" or "my grandma" it will output "she/her". 

# In[ ]:


####### THE FOLLOWING ARE EXPERIMENTS


# In[64]:


learn.predict('20', n_words=9)


# In[705]:


learn.predict('one', n_words=9)


# In[70]:


learn.predict('white house is', n_words=9)


# In[29]:


learn.predict('Trump said', n_words=8)


# In[30]:


learn.predict('Taylor Swift said', n_words=8)


# ## Q: What aspects of the task of generating headlines does our model do poorly? (5 points)
# What does it frequently get wrong? Why might it make these mistakes?
# 
# 

# # Your answer here
# 
# 1. When we restrict the sentence to predict certain number of words, the classifier has little sense of when it should finish a sentence. I guess because the machanism is to output the next string based on the previous strings, so it only cares if the next string connects well with the previous, without considering what word it shoud use to end the sentence.
# 
# 2. I feel like it does propositions pretty poorly. I think the problem is a word can connect with many different propositions (e.g. "I sit at" or "I sit under" or "I sit on" or "I sit down") but when the context is different, some propositions are appropriate but some are not (e.g. 'I travel on airport first time...')
# 
# 3. It outputs random words like 'xxbos' and random punctuations. e.g. "Taylor Swift ' new ' watched video ' recording studio has" or  "Trump . n't n't to accommodate him to n't rebel" or "Trump 's actions against trump improve ratings for tourists xxbos". In the language model, 'xxbos' stands for the start of a new sentence and the random punctuations come from other tokenization rules (e.g. didn't is parsed into did and n't and hers' might be hers and ', so these punctuations were treated as a separated word). I think having 'xxbos' in the sentences indicates the next word is the first word of a new sentence. But it makes no sense to users on twitter when the chatbot writes a sentence like this. 

# # Step 3: Learn a classifier to see which headlines are satire
# 
# Remember, our dataset has some stories that are satire (from the Onion) and others that are real. Now, we're going to train a classifier to distinguish one from the other. 

# In[73]:


data_clas = (TextList.from_df(df=df, path=data_path, vocab= data_lm.train_ds.vocab, cols='headline').split_by_rand_pct(valid_pct=0.2).label_from_df(cols='is_sarcastic').databunch())


# We're using a similar databunch method as we did for our language model above. Here, we are using `split_by_rand_pct` so we keep some fraction of our dataset as a validation set. There is one other trick: `vocab= data_lm.train_ds.vocab` ensures that our classifier only uses words that we have in our language model -- so it never deals with words it hasn't encountered before. (Consider: why is this important?)
# 
# See if you can work out what the other arguments are. 

# In[74]:


data_clas.show_batch()


# Above: what our data looks like after we apply the vocabulary restriction. `xxunk` is an unknown word. 

# Below: we're creating a classifier. 

# In[75]:


classify = text_classifier_learner(data=data_clas, arch=AWD_LSTM, drop_mult=0.5)


# Remember that language model we saved earlier? It's time load it back!

# In[76]:


classify.load_encoder('headlines-awd.pkl')


# What's happening here? 
# 
# Here's the trick: a language model predicts the next word in a sequence using all the information it has so far (all the previous words). When we train a classifier, we ask it to predict the label (satire or not) instead of the next word. 
# 
# The intuition here is that if you can tell what the next word in a sentence is, you can tell if it is satirical. (Similarly, if you can can tell what the next word in an email is, you can tell if it is spam, etc.)

# In[77]:


classify.fit_one_cycle(1, 1e-2)


# In[78]:


classify.freeze_to(-2)


# Above: this is similar to `unfreeze()` that we used before. Except, you only allow a few layers of your model to change. Then we can train again, similar to using `unfreeze()`

# In[79]:


classify.fit_one_cycle(1, 1e-3)


# Wow! An accuracy of 85%! That sounds great, and for not that much work. 
# 
# Now, let's try it on some headlines, to see how well it does. 

# # Step 4: try out the classifier (20 points)

# In[80]:


classify.predict("Despair for Many and Silver Linings for Some in California Wildfires")


# Here in the output, the first part of this tuple is the chosen category (`0`, i.e. not satire), and the last part is an array of probabilities. The classifier suggests that the headline (which I got from the [New York Times](https://www.nytimes.com/2019/10/29/us/california-fires-homes.html?action=click&module=Top%20Stories&pgtype=Homepage)) is not satire, with about an 86% confidence. 

# ## Step 4a: Try out this classifier (10 points)
# 
# Below, try the classifier with some headlines, real or made up (including made up by the language model above). 
# 

# ## Two headlines that the classifier correctly classifies (1 point)

# In[86]:


# this is a satire that is classified correctly
classify.predict("horrified nurses discover 40-pound baby after accidentally leaving it in incubator over weekend")


# In[87]:


# this is generated by the language model above, correctly classified as not satire
classify.predict("16 Problems of life on the street")


# ## Two headlines that the classifier classifies incorrectly (1 point)

# In[88]:


# This is from the Daily Mash. A man who extended his life span by avoiding processed meat bitterly said it is not worth it
classify.predict("it wasn't worth it, says 103-year-old vegetarian")


# In[89]:


# This is a satire headline I found on twitter. There are only 300 million+ people in the entire United States
classify.predict("taylor swift inspires 200 million fans to register to vote in tennessee")


# Now, we want to find two headlines that the classifier is really confident about, but classifies incorrectly. We want the confidence of the prediction to be at least 85%.
# 
# One headline is anything you want to write. Another must be a real headline (not satire) that you could trick the classifier into misclassifying changing only one word. For instance, taking `"Despair for Many and Silver Linings for Some in California Wildfires"`, a real NYTimes headline, you can change it to `"Despair for Many and Silver Linings for Some in Oregon Wildfires"` (note that this particular change does not cause the classifier to misclassify).

# In[90]:


## Insert one headline that the classifier classifies incorrectly, with false high confidence. (4 points)

## 1.
# this is pretty funny to me (modified from a real headline) but the classifier confidently said this is not a satire
classify.predict("running a marathon adds 30 minutes to your life but takes 180 minutes")


# In[ ]:


## 2.
# the original headline is: i love going to the dentist says psychopath with perfect smile
# source: https://www.thedailymash.co.uk/news/lifestyle/i-love-the-dentist-says-psychopath-with-perfect-smile-20190927189363


# In[371]:


#original headline has 0.8990 confidence it is not a satire
test_text= "i love going to the dentist says psychopath with perfect smile"
classify.predict(test_text)


# In[372]:


txt_ci.show_intrinsic_attention(test_text,cmap=cm.Purples)
txt_ci.intrinsic_attention(test_text)[1]


# In[373]:


#changed headline has 0.4342 confidence it is a satire
test_text= "farmers love going to the dentist says psychopath with perfect smile"
classify.predict(test_text)


# In[355]:


txt_ci.show_intrinsic_attention(test_text,cmap=cm.Purples)
txt_ci.intrinsic_attention(test_text)[1]


# ## Step 4b: What kinds of headlines are misclassified? (10 points)
# 
# Write your hypothesis below on what kinds of headlines are misclassified. If it helps you, use the [TextClassificationInterpretation](https://docs.fast.ai/text.learner.html#TextClassificationInterpretation) utility. Show your work, especially if you use this utility.

# In[112]:


## Show work here
import matplotlib.cm as cm
txt_ci = TextClassificationInterpretation.from_learner(classify)


# # Experiment 1:
# No word besides 'to' can be recognized by the language model, but this structure still makes the classifier thinks this is a satire.

# In[113]:


test_text = "zambia exports metals to ethiopia"
classify.predict(test_text )


# In[114]:


txt_ci.show_intrinsic_attention(test_text,cmap=cm.Purples)
txt_ci.intrinsic_attention(test_text)[1]


# # Experiment 2:
# The following series of sentences compare the effects of individual words (farmers vs. I vs. trump; spreading vs. putting). 
# * When the subject is farmer, the classifier thinks the headline is a satire. 
# * When the subject is 'trump', the classifier shifts more attention to 'celebrates' and 'spreading' and become even more certain that it is a satire. 
# * When the subject is 'I', the classifier paid more attention to 'celebrate' but paid less attention to everything else and became 90%+ sure it is not a satire.
# * When I used "putting" instead of "spreading", it shifted more attention to the sentence after 'by' and was less confident that it is a satire.

# In[394]:


'test_text="farmers celebrate spring by spreading shit everywhere"                           
txt_ci.show_intrinsic_attention(test_text,cmap=cm.Purples)
txt_ci.intrinsic_attention(test_text)[1]


# In[395]:


classify.predict(test_text)


# In[392]:


test_text="I celebrate spring by spreading shit everywhere"                           
txt_ci.show_intrinsic_attention(test_text,cmap=cm.Purples)
txt_ci.intrinsic_attention(test_text)[1]


# In[393]:


classify.predict(test_text)


# In[387]:


# change farmer into other words (im just using trump as an example)
test_text= "trump celebrates spring by spreading shit everywhere"
txt_ci.show_intrinsic_attention(test_text,cmap=cm.Purples)
txt_ci.intrinsic_attention(test_text)[1]


# In[388]:


classify.predict(test_text)


# In[389]:


test_text= "trump celebrates spring by putting shit everywhere"
txt_ci.show_intrinsic_attention(test_text,cmap=cm.Purples)
txt_ci.intrinsic_attention(test_text)[1]


# In[390]:


classify.predict(test_text)


# # Experiment 3:
# Using numbers to start the sentence makes the classifier think it is not a satire

# In[380]:


test_text="16 farmers celebrate spring by spreading shit everywhere"                           
txt_ci.show_intrinsic_attention(test_text,cmap=cm.Purples)
txt_ci.intrinsic_attention(test_text)[1]


# In[381]:


classify.predict(test_text)


# In[382]:


test_text="16 things you need to know about sleep"
txt_ci.show_intrinsic_attention(test_text,cmap=cm.Purples)
txt_ci.intrinsic_attention(test_text)[1]


# In[335]:


classify.predict(test_text)


# # Other experiments:
# Change position of hideous to see if classifier still gives it high weight, meaning hideous has the trait of being recognized as a sarcastic word independent of its position

# In[397]:


test_text="even foetus embarrassed by hideous gender reveal party"
txt_ci.show_intrinsic_attention(test_text,cmap=cm.Purples)
txt_ci.intrinsic_attention(test_text)[1]


# In[398]:


classify.predict(test_text)


# In[396]:


test_text="even hideous foetus embarrassed by gender reveal party"
txt_ci.show_intrinsic_attention(test_text,cmap=cm.Purples)
txt_ci.intrinsic_attention(test_text)[1]


# In[333]:


classify.predict(test_text)


# In[299]:


txt_ci.show_top_losses(10)


# # Interpretation notes
# 
# Classification Patterns that I detected: 
# * In general, I found that the classifier gives greater attention to the first couple of words that start the sentence and the last couple of words that ends a sentence. If the first couple of words are the strong words (i.e. words that classifier thinks is likely to be sarcastic, examples include celebrate, hideous, farmers, etc.) then the classifier is more likely to think it is a satire; 
# * Changing a stronger word into a weaker word (e.g. change farmers into trump) will make the classifier shift attention to other strongly charged words. For example, if I change the word "farmer" in headline "farmers celebrates spring by spreading shit everywhere" into "trump", the classifier pays more attention to "spreading" than before (and because it probably thinks spreading is sarcastic, it is more confident that the trump headline is a satire than the farmers headline). 
# 
# # When does the classifier misclassify headlines:
# 
# 1) I feel like there are certain strong words that the classifier tends to determine as a satire (e.g. celebrate, hideous, farmers) and certain words that the classifier thinks is not satire (e.g. I, we). For example, for one of the examiples I tested above, when I use "farmers" as the subject the classifier thinks the headline "farmers celebrate spring by spreading shit everywhere" is a satire; when I change the subject from  "farmers" to "I", it changes the judgement of the classifier (it thinks this is not a satire with 94%+ confidence). I have tried several different examples—so I think if a satires starts with "I" or "we", the classifier is likely to misclassify it as non-satire. If a sober headline has strong words such as "hideous", it is slightly more likely to misclassify it as satire.
# 
# 3) I found that if a headline starts with numbers (e.g. "16 things you need to know"), it is rarely classified as a satire. When I put 'farmers celebrates...' the classifier is almost 50% confident that it is a satire. However, if I add a number such as 16 in front of the sentence ("16 farmers celebrates..."), the classifier is 90%+ confident that it is not a satire—so I think if a satire starts with a number, the classifier is likely to misclassify it as non-satire.
# 

# # Step 5: Save your classifier
# Now that we've trained the classifier, you're ready for Part 2. You'll use this saved file in your bot later.

# In[346]:


classify.export(file='satire_awd.pkl')


# Later, you'll use it like so.

# In[347]:


serve_classifier = load_learner(path=data_path, file='satire_awd.pkl')
serve_lm = load_learner(path=data_path, file='headlines-lm.pkl')


# In[348]:


serve_classifier.predict('How the New Syria Took Shape')


# In[350]:


serve_lm.predict('Rising Seas', n_words=7)


# # Step 6: add the bot code. 
# 
# See the assignment document for what the bot code should look like. You can add it just below here, but you are also welcome to create a new notebook where you put that code. 

# In[351]:


# its in the other notebook!

