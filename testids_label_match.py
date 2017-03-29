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

f=open("count_output_600.txt","r")
lines=f.readlines()
label_dic = defaultdict(lambda:0.0)
topic_dic = defaultdict(lambda:0.0)
nlab=0
ntop=0
for l in lines:
    k=l.split('||')
    k[1]=k[1].strip()
    k[0]=k[0].strip()
    if topic_dic[k[0]]==0:
        ntop+=1
        topic_dic[k[0]]=ntop
    if label_dic[k[1]]==0:
        nlab+=1
        label_dic[k[1]]=nlab
probab=[[0 for x in range(nlab+1)] for y in range(ntop+1)]
for l in lines:
    k=l.split('||')
    k[0]=k[0].strip()
    k[1]=k[1].strip()
    probab[topic_dic[k[0]]][0]=k[0]
    probab[0][label_dic[k[1]]]=k[1]
    probab[topic_dic[k[0]]][label_dic[k[1]]]=k[2]
#for i in range(ntop+1):
    #print probab[i]


dic=defaultdict(lambda:0.0)

for ind in test_leg_ids[:400]:
    dic[ind]=[]
    topics=lda.get_document_topics(corpus[len(legal)+ind])
    for t in topics:
        if str(t[0]) in topic_dic.keys():
            for l in set(labelset):
                if l in label_dic.keys():
                    score=float(probab[topic_dic[str(t[0])]][label_dic[l]])*t[1]
                    if score>0.05:
                        dic[ind].append([l,t[0],score])
    dic[ind] = sorted(dic[ind], key=lambda x: -x[2])
    print dic[ind]
