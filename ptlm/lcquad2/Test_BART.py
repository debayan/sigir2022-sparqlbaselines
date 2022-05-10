from transformers import BartTokenizer, BartForConditionalGeneration

import json
import os

import torch
import pickle
import torch.nn as nn

import re
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--test_file',type=str,default=None)
parser.add_argument('--model_name',type=str,default='t5-base')
parser.add_argument('--checkpoint',type=str,default=None)
parser.add_argument('--device',type=int,default=0)
parser.add_argument('--beam_length',type=int,default=10)
args=parser.parse_args()

torch.manual_seed(42)

file=open(args.test_file,'rb')
final_data_test=pickle.load(file)
file.close()


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

                self.tokenizer=BartTokenizer.from_pretrained(args.model_name)
                self.model=nn.DataParallel(Model(args.model_name),device_ids=[args.device])
                self.model.to(f'cuda:{self.model.device_ids[0]}')  
                
                self.num_gpus=1
                self.eval_bs=8
                self.beam=args.beam_length

                self.args=args
                
                file=open('vocab.txt','r')
                vocab=file.readlines()
                file.close()
                for i in range(len(vocab)):
                        vocab[i]=vocab[i].strip()
                vocab.append('null')
                        
                self.vocab_dict={}
                i=0
                for text in vocab:
                        self.vocab_dict['<extra_id_'+str(i)+'>']=text
                        i+=1
                        
                self.vocab=[]
                i=0
                for text in vocab:
                        self.vocab.append('<extra_id_'+str(i)+'>')
                        i+=1
                        
                self.tokenizer.add_tokens(self.vocab)
                self.model.module.model.resize_token_embeddings(len(self.tokenizer))
                
                params=torch.load(args.checkpoint);
                self.model.load_state_dict(params);
                print('started')

                self.test()
                
        def readable(self,string):
            for key in self.vocab_dict:
                string=string.replace(key,' '+self.vocab_dict[key]+' ')
            string=string.replace('  ',' ')
            vals=string.split()
            
            for k in range(len(vals)):
                if bool(re.match(r'q[0-9]+',vals[k])):
                    vals[k]='Q'+vals[k][1:]
                elif bool(re.match(r'p[0-9]+',vals[k])):
                    vals[k]='P'+vals[k][1:]
                        
            return ' '.join(vals)

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
                            append(self.readable(out[int(k*self.beam+s)].replace('<pad>',''). \
                                          replace('</s>','').replace('<unk>','').replace('<s>','').strip()))
                            
                        saver.append(dict)

                temps=''
                if 'mix' in self.args.test_file:
                    temps='mix_'
                print('Saving to {}'.format(os.getcwd())) 
                file=open('BART_'+temps+'test_result.json','w')
                json.dump(saver,file)
                file.close()

tester=Test(final_data_test,args)
