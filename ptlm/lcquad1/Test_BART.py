from transformers import BartTokenizer, BartForConditionalGeneration

import json

import torch
import pickle
import torch.nn as nn

import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--split_file',type=str,default=None)
parser.add_argument('--model_name',type=str,default='t5-base')
parser.add_argument('--checkpoint',type=str,default=None)
parser.add_argument('--device',type=int,default=0)
parser.add_argument('--beam_length',type=int,default=10)
parser.add_argument('--save_dir',type=str,default=None)
args=parser.parse_args()

torch.manual_seed(42)

file=open(args.split_file,'rb')
data=pickle.load(file)
file.close()

final_data_test=data[0]

class Model(nn.Module):
        def __init__(self,model_name):
                super(Model,self).__init__()
                self.model=BartForConditionalGeneration.from_pretrained(model_name)

        def forward(self,input):
                outputs=self.model(input_ids=input['input_ids'], \
                                           labels=input['labels'],  \
                                           attention_mask=input['attention_mask'], \
                                           output_hidden_states=True,output_attentions=True)

                return outputs.loss
                

class Test:
        def __init__(self,data_test,args):
                self.test_data=data_test
                self.split=args.split_file.split('.')[0][-1]
                if 'mix' in args.split_file:
                    self.split='_mix'+self.split

                self.tokenizer=BartTokenizer.from_pretrained(args.model_name)

                self.model=nn.DataParallel(Model(args.model_name),device_ids=[args.device])
                self.model.to(f'cuda:{self.model.device_ids[0]}')  
                
                self.vocab=['<extra_id_'+str(i)+'>' for i in range(18)]
                self.tokenizer.add_tokens(self.vocab)
                self.model.module.model.resize_token_embeddings(len(self.tokenizer))

                self.num_gpus=1
                self.eval_bs=8
                self.beam=args.beam_length

                self.agrs=args
                
                params=torch.load(args.save_dir+'/'+args.checkpoint);
                self.model.load_state_dict(params);
                print('started')
                
                 
                self.sparql_vocab=['?x','{','}','?uri','SELECT', 'DISTINCT', 'COUNT', '(', ')',  \
              'WHERE', '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>', \
              '.','ASK','[DEF]','<http://dbpedia.org/ontology/','<http://dbpedia.org/property/', \
              '<http://dbpedia.org/resource/']
                self.vocab_dict={}
                for i in range(len(self.sparql_vocab)):
                    self.vocab_dict['<extra_id_'+str(i)+'>']=self.sparql_vocab[i]
                self.vocab_dict['<extra_id_17>']=''
                
                self.test()
                
        def readable(self,string):
            for key in self.vocab_dict:
                string=string.replace(key,' '+self.vocab_dict[key]+' ')
            string=string.replace('  ',' ')
            vals=string.split()
                
            for i,val in enumerate(vals):
                if val=='<http://dbpedia.org/ontology/' or val=='<http://dbpedia.org/property/'  \
                or val=='<http://dbpedia.org/resource/':
                    if i<len(vals)-1:
                        vals[i]=val+vals[i+1]+'>'
                        vals[i+1]=''
                        
            return ' '.join(vals).strip().replace('  ',' ')

        def preprocess_function(self,inputs, targets):
                model_inputs=self.tokenizer(inputs, padding=True, \
                                            return_tensors='pt',max_length=512, truncation=True)
                labels=self.tokenizer(targets,padding=True,max_length=512, truncation=True)

                if True:
                    labels["input_ids"] = [
                        [(l if l != self.tokenizer.pad_token_id else -100) \
                         for l in label] for label in labels["input_ids"]
                    ]
                labels['input_ids']=torch.tensor(labels['input_ids'])
                model_inputs["labels"]=labels["input_ids"].to(f'cuda:{self.model.device_ids[0]}')
                model_inputs["input_ids"]=model_inputs["input_ids"].to(f'cuda:{self.model.device_ids[0]}')
                model_inputs["attention_mask"]=model_inputs["attention_mask"].to(f'cuda:{self.model.device_ids[0]}')

                return model_inputs
                
        def test(self):
                self.model.eval()
                bs,i=self.eval_bs,0
                saver=[]
               
                while i<len(self.test_data):
                    bs_=min(bs,len(self.test_data)-i)
                    i+=bs_
                    inp,label=[],[]
                    for j in range(i-bs_,i):
                            inp.append(self.test_data[j][0])
                            label.append(self.test_data[j][1])

                    input=self.preprocess_function(inp,label)
                    
                    output=self.model.module.model.generate(input_ids=input['input_ids'],
                                          num_beams=self.beam,attention_mask=input['attention_mask'], \
                                            early_stopping=True, max_length=100,num_return_sequences=self.beam)
                    
                    out=self.tokenizer.batch_decode(output,skip_special_tokens=False)

                    for k in range(len(out)//self.beam):
                        dict={}
                        dict['question']=self.readable(inp[k])
                        dict['gold_sparql']=self.readable(label[k].strip())
                        dict['top_'+str(self.beam)+'_output']=[]
                        for s in range(self.beam):
                            dict['top_'+str(self.beam)+'_output']. \
                            append(self.readable(out[int(k*self.beam+s)].replace('<pad>','').replace('</s>','').replace('<s>','').replace('<unk>','').strip()))
                            
                        saver.append(dict)
                
                file=open(self.args.save_dir+'/split'+str(self.split)+'_test_result.json','w')
                json.dump(saver,file)
                file.close()

tester=Test(final_data_test,args)

