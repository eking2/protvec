import torch
import pandas as pd
import torch.nn as nn

class skip_gram(nn.Module):

    def __init__(self, vocab_size, embed_size, device, negative_probs='preprocessed/negatives.csv'):

        super().__init__()

        self.emb_size = embed_size
        self.device = device

        # v = embedding for center (input word)
        # u = embedding for context/negative (output words)
        self.in_emb = nn.Embedding(vocab_size, embed_size)
        self.out_emb = nn.Embedding(vocab_size, embed_size)

        # init with uniform distribution
        self.in_emb.weight.data.uniform_(-1, 1)
        self.out_emb.weight.data.uniform_(-1, 1)

        # negative sampling probabilities
        df = pd.read_csv(negative_probs)
        self.dist = torch.tensor(df['prob'].values)


    def sample_neg(self, n_context, neg_samples):

        n_samples = n_context * neg_samples

        # raw labels
        negatives = torch.multinomial(self.dist, n_samples, replacement = True).to(self.device)

        # embed
        # [n_context, neg_samples, emb_size]
        u_neg = self.out_emb(negatives).view(n_context, neg_samples, self.emb_size)

        return u_neg


    def forward(self, centers, contexts):

        # centers = [1, n_contexts]
        # contexts = [1, n_contexts]

        # input
        v = self.in_emb(centers).squeeze(0)

        # outputs
        u_pos = self.out_emb(contexts).squeeze(0)

        # v = [n_contexts, emb_size]
        # u_pos = [n_contexts, emb_size]

        return v, u_pos

    def neg_sample_loss(self, v, u_pos, u_neg):

        n_contexts = v.shape[0]

        # reshape input v to [n_contexts, emb_size, 1]
        v = v.view(n_contexts, self.emb_size, 1)

        # reshape output positive u_pos [n_contexts, 1, emb_size]
        u_pos = u_pos.view(n_contexts, 1, self.emb_size)

        # batch matrix mult
        pos_loss = torch.bmm(u_pos, v).sigmoid().log()  # [n_contexts, 1, 1]
        pos_loss = pos_loss.squeeze()  # [n_contexts]

        # cumulative sum loss over all neg sampled words
        neg_loss = torch.bmm(u_neg.neg(), v).sigmoid().log()  # [n_contexts, neg_samples, 1]
        neg_loss = neg_loss.squeeze().sum(dim=1)  # [n_contexts]

        return -(pos_loss + neg_loss).mean()
