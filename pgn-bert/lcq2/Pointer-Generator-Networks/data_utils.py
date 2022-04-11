import torch
import collections
import json
import sys
from enum_type import SpecialTokens
from random import randrange


def load_single_data(inputstring, inputvector, max_length):
    source_text = []
    target_text = [['wrong taget text']]
    source_vector = []
    words = [x for x in inputstring]
    vector = [[-1]+x[0][1:] for x in inputvector]
    source_text.append(words[:max_length])
    source_vector.append(vector[:max_length])
    return source_text,target_text,source_vector

def load_data(dataset_path, max_length):
    with open(dataset_path, "r", encoding='utf-8') as fin:
        source_text = []
        target_text = []
        source_vector = []
        for line in fin:
            line = line.strip()
            d = json.loads(line)
            words = [x.lower() for x in d[2]]
            sparql = d[6].replace('(',' ( ').replace(')',' ) ').replace('{',' { ').replace('}',' } ').replace("'"," ' ").lower()
            print(sparql)
#            newvars = ['?vr0','?vr1','?vr2','?vr3','?vr4','?vr5']
#            sparql_split = sparql.split()
#            variables = set([x for x in sparql_split if x[0] == '?'])
#            print(variables)
#            for idx,var in enumerate(sorted(variables)):
#                if var == '?maskvar1':
#                    continue         
#                sparql = sparql.replace(var,newvars[idx])
#            print(sparql)
            vector = d[3]
            if len(words) == len(vector): 
                source_text.append(words[:max_length])
                source_vector.append(vector[:max_length])
                target_text.append([x for x in sparql.split()[:max_length]])
            else:
                print("ERROR:",words)
                continue
    return source_text,target_text,source_vector

def build_vocab(text, max_vocab_size, special_token_list):
    word_list = list()
    for word in text:
        word_list.append(word)

    token_count = [(count, token) for token, count in collections.Counter(word_list).items()]
    token_count.sort(reverse=True)
    tokens = [word for count, word in token_count]
    tokens = special_token_list + tokens
    tokens = tokens[:max_vocab_size]

    max_vocab_size = len(tokens)
    idx2token = dict(zip(range(max_vocab_size), tokens))
    token2idx = dict(zip(tokens, range(max_vocab_size)))
    return idx2token, token2idx, max_vocab_size

def text2idx(source_text, target_text, source_vector, token2idx, is_gen=False):
    data_dict = {'source_idx': [], 'source_length': [], 'source_vector': [],
                 'input_target_idx': [], 'output_target_idx': [], 'target_length': [], 'source_sent': [], 'target_sent': [] }

    if is_gen:
        data_dict['extended_source_idx'] = []
        data_dict['oovs'] = []

    sos_idx = token2idx[SpecialTokens.SOS]
    eos_idx = token2idx[SpecialTokens.EOS]
    unknown_idx = token2idx[SpecialTokens.UNK]

    for source_sent, target_sent in zip(source_text, target_text):
        source_idx = [token2idx.get(word, unknown_idx) for word in source_sent]
        input_target_idx = [sos_idx] + [token2idx.get(word, unknown_idx) for word in target_sent]

        if is_gen:
            extended_source_idx, oovs = article2ids(source_sent, token2idx, unknown_idx)
            output_target_idx = abstract2ids(target_sent, oovs, token2idx, unknown_idx) + [eos_idx]
            data_dict['extended_source_idx'].append(extended_source_idx)
            data_dict['oovs'].append(oovs)
        else:
            output_target_idx = [token2idx.get(word, unknown_idx) for word in target_sent] + [eos_idx]

        data_dict['source_idx'].append(source_idx)
        data_dict['source_length'].append(len(source_idx))
        data_dict['source_vector'].append(source_vector)
        data_dict['input_target_idx'].append(input_target_idx)
        data_dict['output_target_idx'].append(output_target_idx)
        data_dict['target_length'].append(len(input_target_idx))
        data_dict['source_sent'].append(source_sent)
        data_dict['target_sent'].append(target_sent)

    return data_dict


def article2ids(article_words, token2idx, unknown_idx):
    ids = []
    oovs = []
    for w in article_words:
        i = token2idx.get(w, unknown_idx)
        if i == unknown_idx:
            if w not in oovs:
                oovs.append(w)
            oov_num = oovs.index(w)
            ids.append(len(token2idx) + oov_num)
        else:
            ids.append(i)
    return ids, oovs


def abstract2ids(abstract_words, article_oovs, token2idx, unknown_idx):
    ids = []
    for w in abstract_words:
        i = token2idx.get(w, unknown_idx)
        if i == unknown_idx:
            if w in article_oovs:
                vocab_idx = len(token2idx) + article_oovs.index(w)
                ids.append(vocab_idx)
            else:
                ids.append(unknown_idx)
        else:
            ids.append(i)
    return ids


def pad_sequence(idx, length, padding_idx):
    max_length = max(length)
    new_idx = []
    for sent_idx, sent_length in zip(idx, length):
        new_idx.append(sent_idx + [padding_idx] * (max_length - sent_length))
    new_idx = torch.LongTensor(new_idx)
    length = torch.LongTensor(length)
    return new_idx, length

def pad_sequence_vector(idx, length):
    max_length = max(length)
    new_idx = []
    for sent_idx, sent_length in zip(idx, length):
        new_idx.append(sent_idx + [968*[-2.0]] * (max_length - sent_length))
    new_idx = torch.FloatTensor(new_idx)
    length = torch.LongTensor(length)
    return new_idx, length


def get_extra_zeros(oovs):
    max_oovs_num = max([len(_) for _ in oovs])
    extra_zeros = torch.zeros(len(oovs), max_oovs_num)
    return extra_zeros
