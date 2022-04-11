import sys,os,json
import requests
from elasticsearch import Elasticsearch
import numpy as np
import random
from elasticsearch import Elasticsearch

es = Elasticsearch(hosts=['http://ltcpu1:49158'])

reldict = json.loads(open('en1.json').read())

def getlabel(entid):
    entid = entid.upper()
    res = es.search(index="wikidataentitylabelindex02", body={"query":{"term":{"uri":{"value":entid}}}})
    if len(res['hits']['hits']) == 0:
        return None
    else:
        return res['hits']['hits'][0]['_source']['wikidataLabel']

def getrellabel(rel):
    rel = rel.upper()
    if rel in reldict:
        return reldict[rel]
    else:
        return None

def addlabels(source):
    newsource = []
    for token in source:
        newsource.append(token)
        if 'wdt:' in token:
            lab = getrellabel(token[4:])
            if lab:
                newsource.append(lab) 
        if 'wd:' in token:
            lab = getlabel(token[3:])
            if lab:
                newsource.append(lab)

        if 'ps:' in token:
            lab = getrellabel(token[3:])
            if lab:
                newsource.append(lab)
        if 'p:' in token:
            lab = getrellabel(token[3:])
            if lab:
                newsource.append(lab)
        if 'pq:' in token:
            lab = getrellabel(token[3:])
            if lab:
                newsource.append(lab)
    return newsource

d = json.loads(open(sys.argv[1]).read())
f = open(sys.argv[2],'w')
for q in d:
    for item in q:
        source = addlabels(item['source'])
        ques = ' '.join(source)
        text = ques + ' [SEP] ' + item['query'] + ' [SEP] ' + json.dumps(item['result'])
        text = text.split()[:512]
        text = ' '.join(text)
        print(ques)
        print(text)
        print(item['correct'])
        f.write(json.dumps({'text':text,'label':int(item['correct'])})+'\n')
f.close()
