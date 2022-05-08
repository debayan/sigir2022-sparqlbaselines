import math
import torch
import random

from data_utils import pad_sequence, pad_sequence_vector, get_extra_zeros, article2ids


class Dataloader:
    def __init__(self, name, config, dataset, batch_size, shuffle=False, drop_last=True):
        self.name = name
        self.config = config
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.device = config['device']
        if self.name == 'train':
            self.iters_per_epoch = config['iters_per_epoch']

        self.step = batch_size
        self.pr = 0
        self.std_pr = 0
        self.pr_end = len(self.target_text)

    def __getattr__(self, name):
        value = getattr(self.dataset, name)
        if value is not None:
            return value
        return None

    def __len__(self):
        if self.name == 'train':
            return self.iters_per_epoch
        else:
            return math.floor(self.pr_end / self.batch_size) if self.drop_last \
                else math.ceil(self.pr_end / self.batch_size)

    def __iter__(self):
        if self.shuffle:
            self._shuffle()
        return self

    def __next__(self):
        if (self.drop_last and self.std_pr + self.batch_size >= self.pr_end) or \
                (not self.drop_last and self.pr >= self.pr_end):
            self.pr = 0
            self.std_pr = 0
            raise StopIteration()

        if self.name == 'train':
            if self.std_pr == self.iters_per_epoch * self.batch_size:  # 3200
                self.pr = 0
                self.std_pr = 0
                raise StopIteration()

        next_batch = self._next_batch_data()
        self.pr += self.batch_size
        self.std_pr += self.batch_size
        return next_batch

    def _shuffle(self):
        keys = []
        values = []

        for key in self.train_data.keys():
            keys.append(key)
            values.append(getattr(self, key))

        values = list(zip(*values))
        random.shuffle(values)
        for key, value in zip(keys, list(zip(*values))):
            getattr(self.dataset, key)[:] = value

    def _next_batch_data(self):
        source_idx = self.source_idx[self.pr:self.pr + self.step]
        source_length = self.source_length[self.pr:self.pr + self.step]
        source_idx, source_length = pad_sequence(source_idx, source_length, self.padding_token_idx)
        #sys.exit(1)
        source_vector,source_vector_length = pad_sequence_vector(self.source_vector[self.pr:self.pr + self.step], source_length)

        input_target_idx = self.input_target_idx[self.pr:self.pr + self.step]
        output_target_idx = self.output_target_idx[self.pr:self.pr + self.step]
        target_length = self.target_length[self.pr:self.pr + self.step]

        input_target_idx, target_length = pad_sequence(input_target_idx, target_length, self.padding_token_idx)
        output_target_idx, _ = pad_sequence(output_target_idx, target_length, self.padding_token_idx)

        batch_data = {
            'source_idx': source_idx.to(self.device),
            'source_vector': source_vector.to(self.device),
            'source_length': source_length.to(self.device),
            'input_target_idx': input_target_idx.to(self.device),
            'output_target_idx': output_target_idx.to(self.device),
            'target_length': target_length.to(self.device)
        }

        if self.is_pgen:
            extended_source_idx = self.extended_source_idx[self.pr:self.pr + self.step]
            extended_source_idx, _ = pad_sequence(extended_source_idx, source_length, self.padding_token_idx)
            oovs = self.oovs[self.pr:self.pr + self.step]
            extra_zeros = get_extra_zeros(oovs)

            batch_data['extended_source_idx'] = extended_source_idx.to(self.device)
            batch_data['oovs'] = oovs
            batch_data['extra_zeros'] = extra_zeros.to(self.device)

        return batch_data

    def get_reference(self):
        return self.source_text,self.target_text

    def interface(self, sentence):
        source_text = sentence.strip().lower().split()
        source_idx = torch.LongTensor([[self.token2idx.get(w, self.unknown_token_idx) for w in source_text]])
        source_length = torch.LongTensor([len(source_text)])

        example = {
            'source_idx': source_idx.to(self.device),  # 1 x src_len
            'source_length': source_length.to(self.device),  # 1
        }

        if self.is_pgen:
            extended_source_idx, oovs = article2ids(source_text, self.token2idx, self.unknown_token_idx)
            extended_source_idx = torch.LongTensor([extended_source_idx])
            oovs = [oovs]
            extra_zeros = get_extra_zeros(oovs)

            example['extended_source_idx'] = extended_source_idx.to(self.device),  # 1 x src_len
            example['oovs'] = oovs
            example['extra_zeros'] = extra_zeros.to(self.device)  # 1 x max_oovs_num

        return example
