#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import codecs
import os
from nltk.corpus import stopwords
import logging, gensim, bz2
from collections import defaultdict


stop = set(stopwords.words('english'))

def removestop(l,s):
	out = """"""
	l = l.strip().split()
	for word in l:
		if word.lower() not in s:
			out += word.lower() + ' '
	return out.strip()

# Print messages or not?
verbose=True

# Read data
with codecs.open("catholic.txt", 'r', 'utf-8') as inp:
	cath_lines = inp.readlines()

with codecs.open("legal2.txt", 'r', 'utf-8') as inp:
	leg_lines = inp.readlines()

# Preprocess catholic texts
cath = []
train_cath_ids = []
test_cath_ids = []
for i,l in enumerate(cath_lines[1:]):
	l = l.strip().split('\t')
	# If they already have labels...
	if len(l) == 3:
		sid, text, labels = l
		# ... add them to the train ids ...
		train_cath_ids.append(i)
	else:
		sid, text = l
		labels = ''
		# ... else add the to the ids that need labeling
		test_cath_ids.append(i)
	# Remove stop words
	st = removestop(text, stop)
	cath.append((sid,st,labels))

if verbose:
	print "Catholic data"
	print "\tTrain: ", len(train_cath_ids)
	print "\tTest: ", len(test_cath_ids)

#Preprocess Legal data
legal = []
train_leg_ids = []
test_leg_ids = []
for i,l in enumerate(leg_lines[1:]):
	l = l.strip().split('\t')
	# If they already have labels...
	if len(l) == 3:
		sid, text, labels = l
		# ... add them to the train ids ...
		train_leg_ids.append(i)
	else:
		sid, text = l
		labels = ''
		# ... else add the to the ids that need labeling
		test_leg_ids.append(i)
	# Remove stop words
	st = removestop(text, stop)
	legal.append((sid,st,labels))

if verbose:
	print "Legal data"
	print "\tTrain: ", len(train_leg_ids)
	print "\tTest: ", len(test_leg_ids), "\n"


texts = [[word for word in doc.lower().split()] for _,doc,_ in legal] + [[word for word in doc.lower().split()] for _,doc,_ in cath]
lda = gensim.models.ldamodel.LdaModel.load("models/50.all.lda.model")
dictionary = gensim.corpora.Dictionary.load("dicts/50.all.stop.dict")

corpus = [dictionary.doc2bow(text) for text in texts]

if verbose:
	print len(dictionary)
	print texts[0], corpus[0]

top = 1
count_dic = defaultdict(lambda:0.0)
labelset = []

for ind in train_leg_ids[:400]:
	labs = legal[ind][2]
	topics = lda.get_document_topics(corpus[ind])
	topics = sorted(topics, key=lambda x: -x[1])
	if verbose:
		print labs, topics, '\n'
	for l in labs.split('||'):
		labelset.append(l)
		for t in topics[:top]:
			count_dic[t[0],l] += 1 * t[1]
			count_dic[l] += 1
			count_dic[t[0]] += 1

for ind in train_cath_ids[:400]:
	labs = cath[ind][2]
	topics = lda.get_document_topics(corpus[len(legal)+ind])
	topics = sorted(topics, key=lambda x: -x[1])
	if verbose:
		print labs, topics, '\n'
	for l in labs.split('||'):
		labelset.append(l)
		for t in topics[:top]:
			count_dic[t[0],l] += 1
			count_dic[l] += 1
			count_dic[t[0]] += 1


for t in range(50):
	for l in set(labelset):
		if count_dic[t,l] > 0:
			probability = count_dic[t,l]/count_dic[l]
			print t, '||', l,'||', probability




# Try several numbers of topics
for t in [50]:
	# Do LDA (takes a while!)
	lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=t, update_every=1, chunksize=500, passes=5)

	if not verbose:
		print "\nAll topics"
		for i in lda.print_topics(30,10):
			print i

		print "\nDocument Topics"
		for i in range(10):
			print lda.get_document_topics(corpus[i])

		print "\nTopic terms"
		for i in range(10):
			print lda.get_topic_terms(i)

	# Save the model and the dictionaries
	if not os.path.exists("models"):
		os.mkdir("models")
	if not os.path.exists("dicts"):
		os.mkdir("dicts")

	fname = "models/"+str(t)+".all.lda.model"
	dfname = "dicts/"+str(t)+".all.stop.dict"
	lda.save(fname)
	dictionary.save(dfname)
	if verbose:
		print "Model and dictionary saved"
