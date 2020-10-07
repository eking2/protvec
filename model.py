import torch
import torch.nn as nn

class skip_gram(nn.Module):

    def __init__(self, vocab_size, embed_size):

        super().__init__()

        self.emb_size = embed_size

        # v = embedding for center (input word)
        # u = embedding for context/negative (output words)
        self.in_emb = nn.Embedding(vocab_size, embed_size)
        self.out_emb = nn.Embedding(vocab_size, embed_size)

        # init with uniform distribution
        self.in_emb.weight.data.uniform_(-1, 1)
        self.out_emb.weight.data.uniform_(-1, 1)

    def forward(self, centers, contexts_negatives):

        # centers = [bs, 1]
        # contexts_negatives = [bs, seq (depends on num of pos and negs)]

        v = self.in_emb(centers)
        u = self.out_emb(contexts_negatives)

        # v = [bs, 1, emb_size]
        # u = [bs, seq, emb_size]

        # reshape for batch matrix mult
        bs = u.shape[0]
        u = u.view(bs, self.emb_size, -1)

        # u = [bs, emb_size, seq]

        pred = torch.bmm(v, u)

        # pred = [bs, 1, seq]

        return pred
