import sys,os,json,re,copy,requests
import json
import argparse

def setup():
    parser=argparse.ArgumentParser()
    parser.add_argument('--save_dir',type=str,default=None)
    args=parser.parse_args()
    return args

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
        url = 'http://134.100.15.203:8892/sparql/'
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

def querymatch(target, predictions):
    target = target.replace('( ?uri )','(?uri)')
    resulttarget = hitkg(target,'target')
    for l,prediction in enumerate(predictions):
        prediction = prediction.replace('( ?uri )','(?uri)')
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

def split_acc(file_name,i,accuracy,mrr,total):
    print('------------------------------------')
    print('Starting split '+str(i))
    file=open(file_name,'r')
    data=json.load(file)
    file.close()
    
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
            print('\n\n')
            
        else: 
            print('NO MATCH')
            print(question['question'].split('[DEF]')[0].strip())
            print('\n\n')
            
        
    print('\n\n\n\n\n')
    return accuracy,mrr,total

if __name__=='__main__':
    accuracy=0
    total=0
    mrr=0

    args=setup()
    print('Prediction Followed by Target\n\n')
    for i in range(1,6):
        accuracy,mrr,total=split_acc(args.save_dir+'/split'+str(i)+'_test_result.json',i,accuracy,mrr,total)
        
#    print(accuracy/5)
