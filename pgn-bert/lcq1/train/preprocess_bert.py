import sys,os,json

d = json.loads(open(sys.argv[1]).read())
f = open(sys.argv[2],'w')
for item in d:
     ques = ' '.join(item['source'])
     for neq in item['nonemptyqueries']:
         text = ques + ' [SEP] ' + neq['query'] + ' [SEP] ' + json.dumps(neq['result'])
         text = text.split()[:512]
         text = ' '.join(text)
         print(ques)
         print(text)
         print(neq['match'])
         f.write(json.dumps({'text':text,'label':int(neq['match'])})+'\n')
f.close()
