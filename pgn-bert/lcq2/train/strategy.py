import torch


def greedy_search(vocab_dist):
    return vocab_dist.view(-1).argmax().item()


class Beam_Search:
    def __init__(
            self,
            beam_size,
            sos_token_idx,
            eos_token_idx,
            unknown_token_idx,
            device,
            idx2token,
            is_attention=False,
            is_pgen=False,
            is_coverage=False
    ):
        self.beam_size = beam_size
        self.sos_token_idx = sos_token_idx
        self.eos_token_idx = eos_token_idx
        self.unknown_token_idx = unknown_token_idx
        self.device = device
        self.idx2token = idx2token
        self.vocab_size = len(idx2token)

        self.is_attention = is_attention
        self.is_pgen = is_pgen
        self.is_coverage = is_coverage

        self.hypothetic_token_idx = [[sos_token_idx]]
        self.hypothetic_token = [[idx2token[sos_token_idx]]]
        self.completed_hypotheses = []
        self.hyp_scores = torch.zeros(1).to(device)

    def stop(self):
        return len(self.completed_hypotheses) == self.beam_size

    def generate(self):
        if len(self.completed_hypotheses) == 0:
            return self.hypothetic_token[0][1:]
        else:
            #return max(self.completed_hypotheses, key=lambda hyp: hyp[1])[0]
            return self.completed_hypotheses

    def step(self, gen_idx, vocab_dists, decoder_hidden_states, kwargs=None):
        vocab_dists = torch.log(vocab_dists.squeeze(1))
        vocab_size = vocab_dists.shape[-1]

        live_hyp_num = self.beam_size - len(self.completed_hypotheses)
        tmp_hyp_scores = (self.hyp_scores.unsqueeze(1).expand_as(vocab_dists) + vocab_dists).view(-1)
        top_scores, top_pos = torch.topk(tmp_hyp_scores, k=live_hyp_num)

        hyp_ids = (top_pos // vocab_size).tolist()
        word_ids = (top_pos % vocab_size).tolist()

        new_idx_hypotheses = []
        new_token_hypotheses = []
        new_ids = []
        new_scores = []

        for hyp_id, word_id, score in zip(hyp_ids, word_ids, top_scores):
            if word_id >= self.vocab_size:
                token = kwargs['oovs'][word_id - self.vocab_size]
                word_id = self.unknown_token_idx
            else:
                token = self.idx2token[word_id]

            new_idx_hyp = self.hypothetic_token_idx[hyp_id] + [word_id]
            new_token_hyp = self.hypothetic_token[hyp_id] + [token]
            if word_id == self.eos_token_idx:
                self.completed_hypotheses.append((new_token_hyp[1:-1], score / (gen_idx - 1)))
            else:
                new_idx_hypotheses.append(new_idx_hyp)
                new_token_hypotheses.append(new_token_hyp)
                new_ids.append(hyp_id)
                new_scores.append(score)

        self.hypothetic_token_idx = new_idx_hypotheses
        self.hypothetic_token = new_token_hypotheses
        self.hyp_scores = torch.tensor(new_scores).to(self.device)

        input_target_idx = torch.LongTensor([[hyp[-1]] for hyp in self.hypothetic_token_idx]).to(self.device)

        decoder_hidden_states = (decoder_hidden_states[0][:, new_ids, :],
                                 decoder_hidden_states[1][:, new_ids, :])

        hyp_num = len(self.hypothetic_token)
        if self.is_attention:
            kwargs['encoder_outputs'] = kwargs['encoder_outputs'][0].repeat(hyp_num, 1, 1)
            kwargs['encoder_masks'] = kwargs['encoder_masks'][0].repeat(hyp_num, 1)
            kwargs['context'] = kwargs['context'][new_ids, :, :]

        if self.is_pgen:
            kwargs['extra_zeros'] = kwargs['extra_zeros'][0].repeat(hyp_num, 1)
            kwargs['extended_source_idx'] = kwargs['extended_source_idx'][0].repeat(hyp_num, 1)

        if self.is_coverage:
            kwargs['coverages'] = kwargs['coverages'][new_ids, :, :]

        return input_target_idx, decoder_hidden_states, kwargs
