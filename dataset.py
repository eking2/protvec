import torch
from torch.utils.data import Dataset
import numpy as np
from collections import Counter
import itertools
from utils import seq_to_kmers, gen_corpus

def pad_seqs(batch):

    '''pad all sequences of contexts_negatives samples to same length, pad with 0s and return mask'''

    centers = [item[0] for item in batch]
    contexts_negatives = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    # pad contexts_negative
    max_len = max([len(c_n) for c_n in contexts_negatives])
    samples = len(centers)
    c_n_padded = np.zeros((samples, max_len))
    labels_padded = np.zeros((samples, max_len))

    # mask with padded
    mask = np.zeros((samples, max_len))

    for i, c_n in enumerate(contexts_negatives):
        seq_len = len(c_n)
        c_n_padded[i, :seq_len] = c_n
        mask[i, :seq_len] = 1
        labels_padded[i, :seq_len] = labels[i]

    centers = torch.tensor(np.stack(centers, axis=0)).unsqueeze(1)
    c_n_padded = torch.tensor(c_n_padded).long()
    labels_padded = torch.tensor(labels_padded)
    mask = torch.tensor(mask)

    return centers, c_n_padded, labels_padded, mask


class Vocab(Dataset):

    def __init__(self, fn, k=3, min_freq=0, window=5, neg_samples=5, subsample_thresh=1e-4):

        self.min_freq = min_freq
        self.window = window
        self.neg_samples = neg_samples
        self.subsample_thresh = subsample_thresh

        self.raw_corpus = gen_corpus(fn, k)  # word tokens
        self.word2idx, self.idx2word = self.corpus_to_mappings()
        self.corpus = self.label_encode()  # numerical
        self.subsampled = self.subsample_corpus()
        self.centers, self.contexts = self.get_center_and_contexts()
        self.negative_samples = self.get_negative_samples()


    def corpus_to_mappings(self):

        '''flatten corpus (list of lists of k-mers) and return word mappings'''

        flat = list(itertools.chain.from_iterable(self.raw_corpus))

        # sort high to low freq
        counter = Counter(flat)
        sorted_vocab = sorted(counter, key=counter.get, reverse=True)

        # drop if count lower than min freq
        word2idx = {word : i for i, word in enumerate(sorted_vocab, 1) if counter[word] >= self.min_freq}
        word2idx['<unk>'] = 0
        idx2word = {i : word for word, i in word2idx.items()}

        return word2idx, idx2word


    def label_encode(self):

        '''convert from amino acid k-mers to numerical'''

        labeled = []
        for seqs in self.raw_corpus:
            labeled.append([self.word2idx.get(aa, self.word2idx['<unk>']) for aa in seqs])

        return labeled


    def subsample_corpus(self):

        '''drop words that appear with high frequency

        P(word_i) = 1 - sqrt(threshold / frequency)'''

        # flatten
        counter = Counter(list(itertools.chain.from_iterable(self.corpus)))
        total = len(counter)
        freqs = {word : count/total for word, count in counter.items()}
        prob_drop = {word : 1 - np.sqrt(self.subsample_thresh / freq) for word, freq in freqs.items()}

        subsampled = [[word for word in seq if np.random.uniform(0, 1) > prob_drop[word]] for seq in self.corpus]

        return subsampled


    def get_center_and_contexts(self):

        '''get center words and context words in surrounding window'''

        centers = []
        contexts = []

        for seq in self.subsampled:
            # each token in seq will be a center
            centers.extend(seq)
            for i in range(len(seq)):

                # random window size
                r_window = np.random.randint(1, self.window + 1)

                # cutoff window if center word is at start or end of sequence
                indices = list(range(max(0, i - r_window),
                                     min(len(seq), i+1+r_window)))

                # ignore center word
                indices.remove(i)

                # save all context words
                contexts.append([seq[idx] for idx in indices])

        return centers, contexts


    def get_negative_samples(self):

        '''select negative samples to update based on frequency'''

        subsampled_vocab = list(itertools.chain.from_iterable(self.subsampled))
        subsampled_counter = Counter(subsampled_vocab)
        weights = np.array([count**0.75 for word, count in subsampled_counter.items()])
        prob = weights / np.sum(weights)

        negatives = []
        for i, center in enumerate(self.centers):
            samples = []
            while len(samples) < self.neg_samples:
                neg = np.random.choice(list(subsampled_counter.keys()), size=1, p=prob)

                # neg is not in contexts and not center
                if neg not in self.contexts[i] and neg != center:
                    samples.append(neg.item())

            negatives.append(samples)

        return negatives

    def __len__(self):

        '''number of center words, each will be paired to positive and negative samples'''

        return len(self.centers)

    def __getitem__(self, idx):

        '''return centers, contexts+negatives, labels (positive or negative pair)'''

        # pad/mask after batching with collate_fn, random batching will result in different sized contexts_negatives

        contexts = self.contexts[idx]
        negatives = self.negative_samples[idx]

        contexts_negatives = np.array([contexts + negatives]).squeeze(0)
        labels = np.concatenate([np.ones(len(contexts)), np.zeros(len(negatives))])

        return np.array(self.centers[idx]), contexts_negatives, labels



#test = Vocab('./inputs/test_seq.txt')
#print(test[0])
#print(test.contexts)
#print(test.centers)
#print(len(test))
#print(len(test.contexts))

