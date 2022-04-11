import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_enc_layers, dropout_ratio, bidirectional=True):
        super(Encoder, self).__init__()
        self.encoder = nn.LSTM(embedding_size, hidden_size, num_enc_layers,
                               batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)

    def forward(self, source_embeddings, source_length):
        packed_source_embeddings = pack_padded_sequence(source_embeddings, source_length.cpu(),
                                                        batch_first=True, enforce_sorted=False)

        encoder_outputs, encoder_hidden_states = self.encoder(packed_source_embeddings)

        encoder_outputs, _ = pad_packed_sequence(encoder_outputs, batch_first=True)

        return encoder_outputs, encoder_hidden_states


class Decoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            embedding_size,
            hidden_size,
            num_dec_layers,
            dropout_ratio=0.0,
            is_attention=False,
            is_pgen=False,
            is_coverage=False
    ):
        super(Decoder, self).__init__()

        self.decoder = nn.LSTM(embedding_size, hidden_size, num_dec_layers,
                               batch_first=True, dropout=dropout_ratio)
        self.vocab_linear = nn.Linear(hidden_size, vocab_size)

        self.is_attention = is_attention
        self.is_pgen = is_pgen and is_attention
        self.is_coverage = is_coverage and is_attention

        context_size = hidden_size

        if self.is_attention:
            self.x_context = nn.Linear(embedding_size + context_size, embedding_size)
            self.attention = Attention(context_size, hidden_size, is_coverage)
            self.attention_dense = nn.Linear(hidden_size + context_size, hidden_size)

        if is_pgen:
            self.p_gen_linear = nn.Linear(context_size + hidden_size + embedding_size, 1)

    def forward(self, input_embeddings, decoder_hidden_states, kwargs=None):
        if not self.is_attention:
            decoder_outputs, decoder_hidden_states = self.decoder(input_embeddings, decoder_hidden_states)
            vocab_dists = F.softmax(self.vocab_linear(decoder_outputs), dim=-1)
            return vocab_dists, decoder_hidden_states, kwargs

        else:
            vocab_dists = []
            encoder_outputs = kwargs['encoder_outputs']
            encoder_masks = kwargs['encoder_masks']
            context = kwargs['context']

            extra_zeros = None
            extended_source_idx = None
            if self.is_pgen:
                extra_zeros = kwargs['extra_zeros']
                extended_source_idx = kwargs['extended_source_idx']

            coverage = None
            if self.is_coverage:
                coverage = kwargs['coverages']
                coverages = []
                attn_dists = []

        dec_length = input_embeddings.size(1)

        for step in range(dec_length):
            step_input_embeddings = input_embeddings[:, step, :].unsqueeze(1)  # B x 1 x 128

            x = self.x_context(torch.cat((step_input_embeddings, context), dim=-1))  # B x 1 x 128

            decoder_outputs, decoder_hidden_states = self.decoder(x, decoder_hidden_states)  # B x 1 x 256

            context, attn_dist, coverage = self.attention(decoder_outputs, encoder_outputs,
                                                          encoder_masks, coverage)  # B x 1 x src_len

            vocab_logits = self.vocab_linear(self.attention_dense(torch.cat((decoder_outputs, context), dim=-1)))
            vocab_dist = F.softmax(vocab_logits, dim=-1)  # B x 1 x vocab_size

            if self.is_pgen:
                p_gen_input = torch.cat((context, decoder_outputs, x), dim=-1)  # B x 1 x (256 + 256 + 128)
                p_gen = torch.sigmoid(self.p_gen_linear(p_gen_input))  # B x 1 x 1
                attn_dist_ = (1 - p_gen) * attn_dist  # B x 1 x src_len

                # B x 1 x (vocab_size+max_oovs_num)
                extended_vocab_dist = torch.cat(((vocab_dist * p_gen), extra_zeros.unsqueeze(1)), dim=-1)

                vocab_dist = extended_vocab_dist.scatter_add(2, extended_source_idx.unsqueeze(1), attn_dist_)

            if self.is_coverage:
                attn_dists.append(attn_dist)
                coverages.append(coverage)

            vocab_dists.append(vocab_dist)

        vocab_dists = torch.cat(vocab_dists, dim=1)  # B x dec_len x vocab_size+(max_oovs_num)

        kwargs['context'] = context

        if self.is_coverage:
            coverages = torch.cat(coverages, dim=1)  # B x dec_len x src_len
            attn_dists = torch.cat(attn_dists, dim=1)  # B x dec_len x src_len
            kwargs['attn_dists'] = attn_dists
            kwargs['coverages'] = coverages

        return vocab_dists, decoder_hidden_states, kwargs


class Attention(nn.Module):
    def __init__(self, source_size, target_size, is_coverage=False):
        super(Attention, self).__init__()
        self.source_size = source_size
        self.target_size = target_size
        self.is_coverage = is_coverage

        if self.is_coverage:
            self.coverage_linear = nn.Linear(1, target_size, bias=False)

        self.energy_linear = nn.Linear(source_size + target_size, target_size)
        self.v = nn.Parameter(torch.rand(target_size, dtype=torch.float32))

    def score(self, decoder_outputs, encoder_outputs, coverages):
        tgt_len = decoder_outputs.size(1)
        src_len = encoder_outputs.size(1)

        # B * tgt_len * src_len * target_size
        decoder_outputs = decoder_outputs.unsqueeze(2).repeat(1, 1, src_len, 1)
        encoder_outputs = encoder_outputs.unsqueeze(1).repeat(1, tgt_len, 1, 1)
        energy = self.energy_linear(torch.cat((decoder_outputs, encoder_outputs), dim=-1))
        if self.is_coverage:
            coverages = self.coverage_linear(coverages.unsqueeze(3))
            energy = energy + coverages
        energy = torch.tanh(energy)
        energy = self.v.mul(energy).sum(dim=-1)
        return energy  # B * tgt_len * src_len

    def forward(self, decoder_outputs, encoder_outputs, encoder_masks, coverages=None):
        tgt_len = decoder_outputs.size(1)
        energy = self.score(decoder_outputs, encoder_outputs, coverages)
        probs = F.softmax(energy, dim=-1) * encoder_masks.unsqueeze(1).repeat(1, tgt_len, 1)
        normalization_factor = probs.sum(-1, keepdim=True) + 1e-12
        probs = probs / normalization_factor
        context = probs.bmm(encoder_outputs)

        if self.is_coverage:
            coverages = probs + coverages

        return context, probs, coverages
