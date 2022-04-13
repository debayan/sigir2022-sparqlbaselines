import csv
import random
import numpy as np
import pickle
import re

file=open('sparql.csv','r')
csvreader = csv.reader(file)
sparql=[]
for row in csvreader: sparql.append(row[1])
file.close()

file=open('labels.pickle','rb')
labels=pickle.load(file)
file.close()

def get_ent_rels(lis_temp):
    joiner=' '+vocab_dict['[DEF]']+' '
    lis_temp=joiner.join(lis_temp)
    lis_temp=lis_temp.replace('<http://dbpedia.org/ontology/',vocab_dict['<http://dbpedia.org/ontology/']+' ')
    lis_temp=lis_temp.replace('<http://dbpedia.org/property/',vocab_dict['<http://dbpedia.org/property/']+' ')
    lis_temp=lis_temp.replace('<http://dbpedia.org/resource/',vocab_dict['<http://dbpedia.org/resource/']+' ')
    temp1=''
    for t in range(len(lis_temp)):
        if lis_temp[t]=='>' and lis_temp[t-10:t-5]!='extra' and lis_temp[t-11:t-6]!='extra': continue
        else: temp1+=lis_temp[t]
        
    return temp1

#ent_rels_list=[]
sparql_vocab=['?x','{','}','?uri','SELECT', 'DISTINCT', 'COUNT', '(', ')',  \
              'WHERE', '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>', \
              '.','ASK','[DEF]','<http://dbpedia.org/ontology/','<http://dbpedia.org/property/', \
              '<http://dbpedia.org/resource/']
vocab_dict={}
for i in range(len(sparql_vocab)):
    vocab_dict[sparql_vocab[i]]='<extra_id_'+str(i)+'>'

ent_rels=[]
ent_rels_shuffle=[]
for i in range(len(sparql)):
    sparql[i]=sparql[i].replace('{', '{ ').replace('}', ' }').replace('COUNT(?uri)', 'COUNT ( ?uri )').replace('?uri.','?uri .')
    text=sparql[i].split()
    for j,item in enumerate(text):
        if item in sparql_vocab:
                text[j]=vocab_dict[item]

    temp=' '.join(text)
    
    lis_temp=re.findall(r'<http://dbpedia.org/[a-z]+/.*?>',temp)
#    ent_rels_list.append(lis_temp)
    for t in range(len(lis_temp)):
        lis_temp[t]=lis_temp[t]+' '+labels[i][t].strip()
    
    ent_rels.append(get_ent_rels(lis_temp))
    random.shuffle(lis_temp)
    ent_rels_shuffle.append(get_ent_rels(lis_temp))
    
    temp=temp.replace('<http://dbpedia.org/ontology/',vocab_dict['<http://dbpedia.org/ontology/']+' ')
    temp=temp.replace('<http://dbpedia.org/property/',vocab_dict['<http://dbpedia.org/property/']+' ')
    temp=temp.replace('<http://dbpedia.org/resource/',vocab_dict['<http://dbpedia.org/resource/']+' ')
    temp1=''
    for t in range(len(temp)):
        if temp[t]=='>' and temp[t-10:t-5]!='extra' and temp[t-11:t-6]!='extra': continue
        else: temp1+=temp[t]
    sparql[i]=temp1
    
file=open('que.csv','r')
csvreader = csv.reader(file)
ques=[]
for row in csvreader: ques.append(row[1].lower())
file.close()

data=[]
for i in range(len(sparql)):
    que=ques[i]
    que_normal=que+' '+vocab_dict['[DEF]']+' '+ent_rels[i]
    que_shuffle=que+' '+vocab_dict['[DEF]']+' '+ent_rels_shuffle[i]
    data.append([que_normal.strip(),que_shuffle.strip(),sparql[i].strip()])
    
random.shuffle(data)

data1=[]
data2=[]
for item in data:
    data1.append([item[0],item[2]])
    data2.append([item[1],item[2]])
    
def split_save(obj,file_name):
    split_data=np.array_split(obj,5)
    for i in range(5):
        split_data[i]=split_data[i].tolist()
    
    file=open(file_name,'wb')
    pickle.dump(split_data,file)
    file.close()

split_save(data2,'split_mix.pickle')
