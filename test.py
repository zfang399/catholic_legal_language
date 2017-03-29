from collections import defaultdict
f=open("testfile.txt","r")
lines=f.readlines()
label_dic = defaultdict(lambda:0.0)
topic_dic = defaultdict(lambda:0.0)
nlab=0
ntop=0
for l in lines:
    k=l.split('||')
    if topic_dic[k[0]]==0:
        ntop+=1
        topic_dic[k[0]]=ntop
    if label_dic[k[1]]==0:
        nlab+=1
        label_dic[k[1]]=nlab
probab=[[0 for x in range(nlab+1)] for y in range(ntop+1)]
for l in lines:
    k=l.split('||')
    probab[topic_dic[k[0]]][0]=k[0]
    probab[0][label_dic[k[1]]]=k[1]
    probab[topic_dic[k[0]]][label_dic[k[1]]]=k[2]
for i in range(ntop+1):
    print probab[i]
