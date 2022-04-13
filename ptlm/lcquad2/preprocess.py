import re
import json
import pickle
import random
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--file_name',type=str,default=None)
args=parser.parse_args()

temp=''
if args.file_name=='test': temp='_KG_4211'

file=open(args.file_name+temp+'.json','r')
data=json.load(file)
file.close()

file=open('lcq2_labels.pickle','rb')
labels=pickle.load(file)
file.close()
labels['quercia']='null'
labels['qui']='null'
labels['}']='null'
labels['p5122'] = 'Ontario public library ID'.lower()
labels['p3888']='Boijmans artist ID'
labels['p5388']='Bulgarian Antarctic Gazetteer ID'
labels['p5151']='Israel Film Fund ID'
labels['p3633']='British Museum place ID'
labels['p1733']='Steam application ID'

file=open('relations.json','r')
rel_labels=json.load(file)
file.close()

file=open('vocab.txt','r')
vocab=file.readlines()
file.close()
for i in range(len(vocab)):
        vocab[i]=vocab[i].strip()
vocab.append('null')
        
vocab_dict={}
i=0
for text in vocab:
        vocab_dict[text]='<extra_id_'+str(i)+'>'
        i+=1
        
for kk in labels:
    if labels[kk] is None: labels[kk]=vocab_dict['null']
        
data_x,data_y=[],[]
data_x_shuffle=[]
for t,inst in enumerate(data):
    wikisparql=inst['sparql_wikidata']
    if inst['question'] is None:
        question=inst['NNQT_question'].replace('{','')
    else:
        question=inst['question'].replace('{','')
    question=question.replace('}','')
        
    lits=re.findall(r"\'(.*?)\'",wikisparql)
    hashi={}
    for idx,elements in enumerate(lits):
       wikisparql=wikisparql.replace("'"+elements.strip()+"'","'###"+str(idx+1)+"'")
       hashi['###'+str(idx+1)]=elements.strip()
        
    sparql = wikisparql.replace('(',' ( ').replace(')',' ) ') \
    .replace('{',' { '). \
    replace('}',' } ').replace('wd:','wd: ').replace('wdt:','wdt: '). \
    replace(' p:',' p: ').replace(' ps:',' ps: ').replace('pq:','pq: '). \
    replace(',',' , ').replace(",'",", '").replace("'"," ' ").replace('.',' . '). \
    replace('=',' = ').replace('  ',' ').lower()
    
    _ents = re.findall( r'wd: (?:.*?) ', sparql)
    _ents_for_labels = re.findall( r'wd: (.*?) ', sparql)
    
    _rels = re.findall( r'wdt: (?:.*?) ',sparql)
    _rels += re.findall( r' p: (?:.*?) ',sparql)
    _rels += re.findall( r' ps: (?:.*?) ',sparql)
    _rels += re.findall( r'pq: (?:.*?) ',sparql)
    
    _rels_for_labels = re.findall( r'wdt: (.*?) ',sparql)
    _rels_for_labels += re.findall( r' p: (.*?) ',sparql)
    _rels_for_labels += re.findall( r' ps: (.*?) ',sparql)
    _rels_for_labels += re.findall( r'pq: (.*?) ',sparql)
    
    for j in range(len(_ents_for_labels)):
#        print('Q'+_ents_for_labels[j][1:])
        if '}' in _ents[j]: 
            _ents[j]=''
        _ents[j]=_ents[j]+labels[_ents_for_labels[j]]+' '
    for j in range(len(_rels_for_labels)):
        if _rels_for_labels[j] not in rel_labels:
            rel_labels['P'+_rels_for_labels[j][1:]]=vocab_dict['null']
        _rels[j]=_rels[j]+rel_labels['P'+_rels_for_labels[j][1:]]+' '
    _ents+=_rels
    random.shuffle(_ents)
#    random.shuffle(_rels)
    
    newvars = ['?vr0','?vr1','?vr2','?vr3','?vr4','?vr5']
    sparql_split = sparql.split()
    variables = set([x for x in sparql_split if x[0] == '?'])
    for idx,var in enumerate(sorted(variables)):
        if var == '?maskvar1':
            continue         
        sparql = sparql.replace(var,newvars[idx])
        
    split=sparql.split()
    for idx, item in enumerate(split):
        if item in vocab_dict:
            split[idx]=vocab_dict[item]
    
    split=' '.join(split).strip()
    
    for keys in hashi:
        split=split.replace(keys,hashi[keys])
    
    data_y.append(split)
#    for ent in _ents:
#            question=question+' '+vocab_dict['[DEF]']+' '+ent
    for rel in _ents:
            rel=rel.replace('wd:',vocab_dict['wd:']+' ')
            rel=rel.replace('wdt:',vocab_dict['wdt:']+' ')
            rel=rel.replace('p:',vocab_dict['p:']+' ')
            rel=rel.replace('ps:',vocab_dict['ps:']+' ')
            rel=rel.replace('pq:',vocab_dict['pq:']+' ')
            question=question+' '+vocab_dict['[DEF]']+' '+rel
    data_x.append(question.strip())
    
data_main=[]
for i in range(len(data_x)):
    data_main.append([data_x[i],data_y[i]])
    
file=open(args.file_name+'_new_mix.pickle','wb')
pickle.dump(data_main,file)
file.close()
