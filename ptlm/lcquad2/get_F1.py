import sys,os,json,re,copy,requests
import json
import argparse
import re

parser=argparse.ArgumentParser()
parser.add_argument('--test_file',type=str,default=None)
args=parser.parse_args()

string_prefix='PREFIX p: <http://www.wikidata.org/prop/> PREFIX pq: <http://www.wikidata.org/prop/qualifier/> PREFIX ps: <http://www.wikidata.org/prop/statement/>   PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wds: <http://www.wikidata.org/entity/statement/> PREFIX wdt: <http://www.wikidata.org/prop/direct/> '

def match(target, answer):
    try:
        tb = target['results']['bindings']
        rb = answer['results']['bindings']
        if tb == rb:
            return True
    except Exception as err:
#        print(err)
        kt=err
    try:
        if target['boolean'] == answer['boolean']:
#            print("boolean true/false match")
            return True
        if target['boolean'] != answer['boolean']:
#            print("boolean true/false mismatch")
            return False 
    except Exception as err:
        return False


def hitkg(query,typeq):
    try:
        url = 'http://ltcpu3:8890/sparql'
#        print(query)
        r = requests.get(url, params={'format': 'json', 'query': query})
        json_format = r.json()
#        print(json_format)
        results = json_format
        if not results and typeq == 'target':
#            print("no response")
            sys.exit(1)
        return results
    except Exception as err:
        #print(err)
        kt=err
        if typeq == 'target':
#            print("no response on target")
            sys.exit(1)
        return ''

def empty(r):
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

def change(string):
    string=string.replace('  ',' ').replace('( ','(').replace(' )',')') \
    .replace('{ ',' {'). \
    replace(' }','}').replace('wd: ','wd:').replace('wdt: ','wdt:'). \
    replace(' p: ',' p:').replace(' ps: ',' ps:').replace('pq: ','pq:'). \
    replace(' , ',', ').replace(", '",",'").replace(" ' ","'").replace("' ","'"). \
    replace(" '","'").replace(' = ', '=').strip()
    
    rep_dec=re.findall('[0-9] \. [0-9]',string)
    for dec in rep_dec:
        string=string.replace(dec,dec.replace(' . ','.'))
    
    return string

def querymatch(target, predictions):
    target=string_prefix+change(target)
    resulttarget = hitkg(target,'target')
    for l,prediction in enumerate(predictions):
        prediction=string_prefix+change(prediction)
        resultanswer = hitkg(prediction,'answer')
        if empty(resultanswer):
            print#("no answer")
            continue
        elif match(resulttarget,resultanswer):
            #print("match")
            return True,target,prediction,l+1
        else :
            #print("match fail")
            return False,None,None,None
    return False,None,None,None

def acc(file_name):
    
    file=open(file_name,'r')
    data=json.load(file)
    file.close()
    accuracy,total,mrr=0,0,0
    
    for question in data:
        temp,target1,prediction1,prediction_no=querymatch(question['gold_sparql'],question['top_10_output'])
        accuracy+=temp
        total+=1
        if target1 is not None:
            mrr+=(1/prediction_no)
            print('MATCH')
            print(question['question'].split('[DEF]')[0].strip())
            print(prediction1)
            print(target1)
            print('Matched rank '+str(prediction_no))
            print('Total number of matches uptil now is :'+str(accuracy))
            print('Accuracy uptil now is: '+str(100*accuracy/total))
            print('MRR uptil now is: '+str(mrr/total))
            print('Total queries uptil now is: '+str(total))
            print('\n\n')
            
        else: 
            print('NO MATCH')
            print(question['question'].split('[DEF]')[0].strip())
            print('Total queries uptil now is: '+str(total))
            print('\n\n')
            
    return


accuracy=0
total=0
mrr=0
print('Prediction Followed by Target\n\n')
    
acc(args.test_file)

