import numpy as np
import codecs
from nltk.corpus import stopwords
import logging, gensim, bz2


stop = set(stopwords.words('english'))

def removestop(l,s):
	out = """"""
	l = l.strip().split()
	for word in l:
		if word.lower() not in s:
			out += word.lower() + ' '
	return out.strip()

# Print messages or not?
verbose=False

# Read data
with codecs.open("catholic.txt", 'r', 'utf-8') as inp:
	cath_lines = inp.readlines()

with codecs.open("legal2.txt", 'r', 'utf-8') as inp:
	leg_lines = inp.readlines()

# Preprocess catholc texts
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

#LDA on both texts
texts = [[word for word in doc.lower().split()] for _,doc,_ in legal] + [[word for word in doc.lower().split()] for _,doc,_ in cath]
dictionary = gensim.corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

if verbose:
	print len(dictionary)
	print texts[0], corpus[0]

# Try several numbers of topics
for t in [50,100,200,300,500,600]:
	# Do LDA (takes a while!)
	lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=t, update_every=1, chunksize=1000, passes=5)

	if verbose:
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
	fname = "models/"+str(t)+".all.lda.model"
	dfname = "dicts/"+str(t)+".all.stop.dict"
	lda.save(fname)
	dictionary.save(dfname)
	if verbose:
		print "Model and dictionary saved"
