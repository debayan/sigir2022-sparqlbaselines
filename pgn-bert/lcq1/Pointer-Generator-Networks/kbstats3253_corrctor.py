import sys,os,json,re,copy,requests
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import load_metric
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np


class KBMatch:
    def __init__(self,chkpath):
        if not chkpath:
            print("No BERT model to load")
            return
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        print("Loading bert classify model from ",chkpath)
        self.model = AutoModelForSequenceClassification.from_pretrained(chkpath, num_labels=2) 
        print("Model loaded.")

    def match(self, target, answer):
        try:
            tb = target['results']['bindings']
            rb = answer['results']['bindings']
            if tb == rb:
                return True
        except Exception as err:
            #print(err)
            pass
        try:
            if target['boolean'] == answer['boolean']:
                print("boolean true/false match")
                return True
            if target['boolean'] != answer['boolean']:
                print("boolean true/false mismatch")
                return False 
        except Exception as err:
            return False
    
    
    def hitkg(self, query,typeq):
        try:
            url = 'http://134.100.15.203:8892/sparql/'
            #print(query)
            r = requests.get(url, params={'format': 'json', 'query': query})
            json_format = r.json()
            #print(json_format)
            results = json_format
            if self.empty(results) and typeq == 'target':
                print("no response in target query")
                #sys.exit(1)
            return results
        except Exception as err:
            #print(err)
            if typeq == 'target':
                print("no response in target query")
                #sys.exit(1)
            return ''
    
    def empty(self,r):
        if not r:
            return True
        if 'boolean' not in r:
            if 'results' in r:
                if 'bindings' in r['results']:
                    if not r['results']['bindings']:
                        return True
                    if {} in r['results']['bindings']:
                        return True
        return False
    
    
    def querymatchbert(self,source, target, predictions):
        print(source)
        target = ' '.join(target)
        target = target.replace('< ','<').replace(' >','>')
        resulttarget = self.hitkg(target,'target')
        #print('target: ',target)
        print(resulttarget)
        nonemptyqueries = []
        unrankedqueries = []
        for prediction in predictions:
            #print('pred2',prediction[0])
            if not prediction[0]:
                continue
            prediction = ' '.join(prediction[0])
            prediction = prediction.replace('< ','<').replace(' >','>')
            resultanswer = self.hitkg(prediction,'answer')
            if not self.empty(resultanswer): #if query returns some answer from KG
                classify_input = ' '.join(source) + ' [SEP] ' + prediction + ' [SEP] ' + json.dumps(resultanswer)
                encoding = self.tokenizer([classify_input], return_tensors="pt", max_length = 512)
                model_outputs = self.model(**encoding)
                unrankedqueries.append({'query':prediction,'result':resultanswer,'score':model_outputs.logits[0][1].item()})
        if len(unrankedqueries) == 0:
            return False
        rankedqueries = sorted(unrankedqueries, key=lambda d: d['score'], reverse=True)
        if self.match(resulttarget,rankedqueries[0]['result']):
            print(rankedqueries[0]['query'])
            print("MATCH")
            return True
        else:
            return False

    def querymatch(self,source, target, predictions):
        print(source)
        target = ' '.join(target)
        target = target.replace('< ','<').replace(' >','>')
        resulttarget = self.hitkg(target,'target')
        #print('target: ',target)
        print(resulttarget)
        nonemptyqueries = []
        unrankedqueries = []
        for prediction in predictions:
            #print('pred2',prediction[0])
            if not prediction[0]:
                continue
            prediction = ' '.join(prediction[0])
            prediction = prediction.replace('< ','<').replace(' >','>')
            resultanswer = self.hitkg(prediction,'answer')
            if not self.empty(resultanswer): #if query returns some answer from KG
                if self.match(resulttarget,resultanswer):
                    return True
                else:
                    return False
        return False
