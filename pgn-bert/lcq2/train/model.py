import os
import torch
import torch.nn as nn

from module import Encoder, Decoder
from strategy import greedy_search, Beam_Search


class AbstractModel(nn.Module):
    def __init__(self, config):
        super(AbstractModel, self).__init__()
        self.config = config
        self.device = config['device']
        self._init_vocab_dict()

    def _init_vocab_dict(self):
        vocab_path = os.path.join(self.config['data_path'], 'vocab.bin')
        self.idx2token, _, self.max_vocab_size = torch.load(vocab_path)
        self.padding_token_idx = 0
        self.unknown_token_idx = 1
        self.sos_token_idx = 2
        self.eos_token_idx = 3


class Model(AbstractModel):
    def __init__(self, config):
        super(Model, self).__init__(config)

        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_enc_layers = config['num_enc_layers']
        self.num_dec_layers = config['num_dec_layers']
        self.bidirectional = config['bidirectional']
        self.dropout_ratio = config['dropout_ratio']
        self.strategy = config['decoding_strategy']
        self.target_max_length = config['tgt_len']

        self.is_attention = config['is_attention']
        self.is_pgen = config['is_pgen'] and self.is_attention
        self.is_coverage = config['is_coverage'] and self.is_attention

        if self.is_coverage:
            self.cov_loss_lambda = config['cov_loss_lambda']

        if self.strategy == 'beam_search':
            self.beam_size = config['beam_size']

        self.context_size = self.hidden_size

        #self.source_token_embedder = nn.Embedding(self.max_vocab_size, self.embedding_size,
        #                                          padding_idx=self.padding_token_idx)
        self.target_token_embedder = nn.Embedding(self.max_vocab_size, self.embedding_size,
                                                  padding_idx=self.padding_token_idx)
        self.encoder = Encoder(
            self.embedding_size, self.hidden_size, self.num_enc_layers,
            self.dropout_ratio, self.bidirectional
        )

        self.decoder = Decoder(
            self.max_vocab_size, self.embedding_size, self.hidden_size,
            self.num_dec_layers, self.dropout_ratio,
            is_attention=self.is_attention,  is_pgen=self.is_pgen, is_coverage=self.is_coverage
        )

    def encode(self, source_idx, source_vector, source_length):
        source_embeddings = source_vector#self.source_token_embedder(source_idx)
        encoder_outputs, encoder_hidden_states = self.encoder(source_embeddings, source_length)

        if self.bidirectional:
            encoder_outputs = encoder_outputs[:, :, self.hidden_size:] + encoder_outputs[:, :, :self.hidden_size]

        encoder_hidden_states = (encoder_hidden_states[0][::2].contiguous(), encoder_hidden_states[1][::2].contiguous())

        return encoder_outputs, encoder_hidden_states

    def generate(self, corpus):
        generated_corpus = []

        source_idx = corpus['source_idx']
        source_vector = corpus['source_vector']
        source_length = corpus['source_length']
        encoder_outputs, encoder_hidden_states = self.encode(source_idx, source_vector, source_length)

        batch_size = len(source_idx)
        src_len = len(source_idx[0])

        for bid in range(batch_size):
            generated_tokens = []

            input_target_idx = torch.LongTensor([[self.sos_token_idx]]).to(self.device)
            decoder_hidden_states = (encoder_hidden_states[0][:, bid, :].unsqueeze(1).contiguous(),
                                     encoder_hidden_states[1][:, bid, :].unsqueeze(1).contiguous())

            kwargs = {}
            if self.is_attention:
                kwargs['encoder_outputs'] = encoder_outputs[bid, :, :].unsqueeze(0)
                kwargs['encoder_masks'] = torch.ne(source_idx[bid], self.padding_token_idx).unsqueeze(0).to(self.device)
                kwargs['context'] = torch.zeros((1, 1, self.context_size)).to(self.device)

            if self.is_pgen:
                kwargs['extra_zeros'] = corpus['extra_zeros'][bid, :].unsqueeze(0)
                kwargs['extended_source_idx'] = corpus['extended_source_idx'][bid, :].unsqueeze(0)
                kwargs['oovs'] = corpus['oovs'][bid]

            if self.is_coverage:
                kwargs['coverages'] = torch.zeros((1, 1, src_len)).to(self.device)

            if self.strategy == 'beam_search':
                hypothesis = Beam_Search(
                    self.beam_size, self.sos_token_idx, self.eos_token_idx, self.unknown_token_idx,
                    self.device, self.idx2token,
                    is_attention=self.is_attention, is_pgen=self.is_pgen, is_coverage=self.is_coverage
                )
            for gen_id in range(self.target_max_length):
                input_embeddings = self.target_token_embedder(input_target_idx)

                vocab_dists, decoder_hidden_states, kwargs = self.decoder(
                    input_embeddings, decoder_hidden_states, kwargs=kwargs
                )

                if self.strategy == 'greedy_search':
                    word_id = greedy_search(vocab_dists)
                    if word_id == self.eos_token_idx:
                        break
                    else:
                        if word_id >= self.max_vocab_size:
                            generated_tokens.append(kwargs['oovs'][word_id - self.max_vocab_size])
                            word_id = self.unknown_token_idx
                        else:
                            generated_tokens.append(self.idx2token[word_id])
                        input_target_idx = torch.LongTensor([[word_id]]).to(self.device)
                elif self.strategy == 'beam_search':
                    input_target_idx, decoder_hidden_states, kwargs = hypothesis.step(
                        gen_id, vocab_dists, decoder_hidden_states, kwargs)
                    if hypothesis.stop():
                        break
            if self.strategy == 'beam_search':
                generated_tokens = hypothesis.generate()
            
            generated_corpus.append(generated_tokens)

        return generated_corpus

    def forward(self, corpus):
        # Encoder
        source_idx = corpus['source_idx']
        source_vector = corpus['source_vector']
        source_length = corpus['source_length']
        encoder_outputs, encoder_hidden_states = self.encode(source_idx, source_vector, source_length)

        batch_size = len(source_idx)
        src_len = len(source_idx[0])

        # Decoder
        input_target_idx = corpus['input_target_idx']
        input_embeddings = self.target_token_embedder(input_target_idx)  # B x dec_len x 128

        kwargs = {}
        if self.is_attention:
            kwargs['encoder_outputs'] = encoder_outputs  # B x src_len x 256
            kwargs['encoder_masks'] = torch.ne(source_idx, self.padding_token_idx).to(self.device)  # B x src_len
            kwargs['context'] = torch.zeros((batch_size, 1, self.context_size)).to(self.device)  # B x 1 x 256

        if self.is_pgen:
            kwargs['extra_zeros'] = corpus['extra_zeros']  # B x max_oovs_num
            kwargs['extended_source_idx'] = corpus['extended_source_idx']  # B x src_len

        if self.is_coverage:
            kwargs['coverages'] = torch.zeros((batch_size, 1, src_len)).to(self.device)  # B x 1 x src_len

        vocab_dists, _, kwargs = self.decoder(
            input_embeddings, encoder_hidden_states, kwargs=kwargs
        )
        # Loss
        output_target_idx = corpus['output_target_idx']
        probs_masks = torch.ne(output_target_idx, self.padding_token_idx)

        gold_probs = torch.gather(vocab_dists, 2, output_target_idx.unsqueeze(2)).squeeze(2)  # B x dec_len
        nll_loss = -torch.log(gold_probs + 1e-12)
        if self.is_coverage:
            coverage_loss = torch.sum(torch.min(kwargs['attn_dists'], kwargs['coverages']), dim=2)  # B x dec_len
            nll_loss = nll_loss + self.cov_loss_lambda * coverage_loss

        loss = nll_loss * probs_masks
        length = corpus['target_length']
        loss = loss.sum(dim=1) / length.float()
        loss = loss.mean()
        return loss
