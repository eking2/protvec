import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from collections import Counter
import itertools
from utils import seq_to_kmers, gen_corpus
from tqdm.auto import tqdm
import json
from pathlib import Path
from joblib import Parallel, delayed

# def pad_seqs(batch):

#     '''pad all sequences of contexts_negatives samples to same length, pad with 0s and return mask'''

#     centers = [item[0] for item in batch]
#     contexts_negatives = [item[1] for item in batch]
#     labels = [item[2] for item in batch]

#     # pad contexts_negative
#     max_len = max([len(c_n) for c_n in contexts_negatives])
#     samples = len(centers)

#     # mask with padded
#     c_n_padded = np.zeros((samples, max_len))
#     labels_padded = np.zeros((samples, max_len))
#     mask = np.zeros((samples, max_len))

#     # fill in data
#     for i, c_n in enumerate(contexts_negatives):
#         seq_len = len(c_n)
#         c_n_padded[i, :seq_len] = c_n
#         mask[i, :seq_len] = 1
#         labels_padded[i, :seq_len] = labels[i]

#     # np array to tensors
#     centers = torch.tensor(np.stack(centers, axis=0)).unsqueeze(1)
#     c_n_padded = torch.tensor(c_n_padded).long()
#     labels_padded = torch.tensor(labels_padded)
#     mask = torch.tensor(mask)

#     return centers, c_n_padded, labels_padded, mask


def concat_seqs(batch):

    '''concat centers, concat contexts, together for each batch'''

    centers = torch.cat([torch.tensor(item[0]) for item in batch])
    contexts = torch.cat([torch.tensor(item[1]) for item in batch])

    # do not need pad
    # centers and contexts are [batch, contexts]

    return centers, contexts

# too large to store in memory
# preprocess vocab and store to text files
# then get samples by byte index
class PreprocessVocab:

    def __init__(self, fn, k=3, min_freq=0, window=5, subsample_thresh=1e-4):

        self.fn = fn
        self.k = k
        self.min_freq = min_freq
        self.window = (window // 2) + 1  # how many behind and forward on window center
        self.subsample_thresh = subsample_thresh


    def corpus_to_mappings(self):

        '''flatten corpus (list of lists of k-mers) and return word mappings'''

        flat = list(itertools.chain.from_iterable(self.raw_corpus))

        # sort high to low freq
        counter = Counter(flat)
        sorted_vocab = sorted(counter, key=counter.get, reverse=True)

        # drop if count lower than min freq
        word2idx = {word : i for i, word in enumerate(tqdm(sorted_vocab), 1) if counter[word] >= self.min_freq}
        word2idx['<unk>'] = 0
        idx2word = {i : word for word, i in word2idx.items()}

        with open('preprocessed/word2idx.json', 'w') as w2i:
            json.dump(word2idx, w2i, indent=2)

        with open('preprocessed/idx2word.json', 'w') as i2w:
            json.dump(idx2word, i2w, indent=2)

        return word2idx, idx2word


    def label_encode(self):

        '''convert from amino acid k-mers to numerical'''

        labeled = []
        for seqs in tqdm(self.raw_corpus):
            labeled.append([self.word2idx.get(aa, self.word2idx['<unk>']) for aa in seqs])

        with open('preprocessed/labeled.txt', 'w') as l:
            for line in labeled:
                l.write(' '.join(map(str, line)))
                l.write('\n')

        return labeled


    def subsample_corpus(self):

        '''drop words that appear with high frequency

        P(word_i) = 1 - sqrt(threshold / frequency)'''

        # flatten
        counter = Counter(list(itertools.chain.from_iterable(self.corpus)))

        # self.corpus is list of lists, want length of all elements in total
        total = len(sum(self.corpus, []))
        freqs = {word : count/total for word, count in counter.items()}
        prob_drop = {word : 1 - np.sqrt(self.subsample_thresh / freq) for word, freq in freqs.items()}

        # deal with negative probabilities from extremely rare kmers
        subsampled = [[word for word in seq if np.random.uniform(0, 1) < (1 - prob_drop[word])] for seq in tqdm(self.corpus)]

        with open('preprocessed/subsampled.txt', 'w') as s:
            for line in subsampled:
                s.write(' '.join(map(str, line)))
                s.write('\n')

        return subsampled, counter


    def get_center_and_contexts(self):

        '''get center words and context words in surrounding window'''

        centers = []
        contexts = []

        for seq in tqdm(self.subsampled):
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

        with open('preprocessed/centers.txt', 'w') as ce, open('preprocessed/contexts.txt', 'w') as co:
            for center, context in zip(centers, contexts):
                ce.write(str(center))
                ce.write('\n')

                co.write(' '.join(map(str, context)))
                co.write('\n')

        return centers, contexts

    # def get_negative_sample(self, i, center, neg_samples, subsampled_counter, prob, contexts):

    #     samples = []
    #     while len(samples) < neg_samples:
    #         neg = np.random.choice(list(subsampled_counter.keys()), size=1, p=prob)

    #         # neg is not in contexts and not center
    #         if neg not in contexts[i] and neg != center:
    #             samples.append(neg.item())

    #     return np.array(samples, dtype=np.int32)

    def get_negative_samples(self):

        '''negative sampling distribution'''

        # merge idx2word and counts
        df = pd.DataFrame.from_dict(self.idx2word, orient='index').reset_index()
        df.columns = ['label', 'kmer']
        counts = pd.DataFrame.from_dict(self.counter, orient='index').reset_index()
        counts.columns = ['label', 'count']
        df = df.merge(counts)

        # calc and output propability for negative sampling
        df['freq'] = df['count']**0.75
        df['prob'] = df['count'] / df['count'].sum()
        df = df.sort_values(by='label')  # move (0 : <unk>) from end to front

#        subsampled_vocab = list(itertools.chain.from_iterable(self.subsampled))
#        subsampled_counter = Counter(subsampled_vocab)
#        weights = np.array([count**0.75 for word, count in subsampled_counter.items()])
#        prob = weights / np.sum(weights)

#        #negatives = Parallel(n_jobs=-1, backend='loky')(delayed(self.get_negative_sample)(i, center, self.neg_samples, subsampled_counter, prob, self.contexts) for i, center in enumerate(tqdm(self.centers)))
#        negatives = np.stack([self.get_negative_sample(i, center, self.neg_samples, subsampled_counter, prob, self.contexts) for i, center in enumerate(tqdm(self.centers))])
#        np.savetxt('preprocessed/negatives.txt', negatives, fmt='%i')

        # with open('preprocessed/negatives.txt', 'w') as n:
        #     for line in negatives:
        #         n.write(' '.join(map(str, line)))
        #         n.write('\n')

        #return negatives
        df.to_csv('preprocessed/negatives.csv', index=False)


    def gensim_setup(self):

        print('tokenizing')
        self.raw_corpus = gen_corpus(self.fn, self.k)  # word tokens
        with open('preprocessed/corpus.txt', 'w') as c:
            for seq in self.raw_corpus:
                c.write(' '.join(seq))
                c.write('\n')

        print('mapping vocab')
        self.word2idx, self.idx2word = self.corpus_to_mappings()

        print('numericalizing')
        self.corpus = self.label_encode()  # numerical


    def run_all(self):

        print('tokenizing')
        self.raw_corpus = gen_corpus(self.fn, self.k)  # word tokens
        with open('preprocessed/corpus.txt', 'w') as c:
            for seq in self.raw_corpus:
                c.write(' '.join(seq))
                c.write('\n')

        print('mapping vocab')
        self.word2idx, self.idx2word = self.corpus_to_mappings()

        print('numericalizing')
        self.corpus = self.label_encode()  # numerical

        print('subsampling')
        self.subsampled, self.counter = self.subsample_corpus()

        print('generating positive pairs')
        self.centers, self.contexts = self.get_center_and_contexts()

        print('generating negative sampling probability')
        self.get_negative_samples()


    # def __len__(self):

    #     '''number of center words, each will be paired to positive and negative samples'''

    #     return len(self.centers)

    # def __getitem__(self, idx):

    #     '''return centers, contexts+negatives, labels (positive or negative pair)'''

    #     # pad/mask after batching with collate_fn, random batching will result in different sized contexts_negatives

    #     contexts = self.contexts[idx]
    #     negatives = self.negative_samples[idx]

    #     contexts_negatives = np.array([contexts + negatives]).squeeze(0)
    #     labels = np.concatenate([np.ones(len(contexts)), np.zeros(len(negatives))])

    #     return np.array(self.centers[idx]), contexts_negatives, labels


class ProtVecVocab(Dataset):

    def __init__(self):

        self.word2idx = json.loads(Path('preprocessed/word2idx.json').read_bytes())
        self.idx2word = json.loads(Path('preprocessed/idx2word.json').read_bytes())
        self.centers_path = 'preprocessed/centers.txt'
        self.contexts_path = 'preprocessed/contexts.txt'

        # dictionary with byte offsets for each line
        self.centers_offsets = self.get_offsets(self.centers_path)
        self.contexts_offsets = self.get_offsets(self.contexts_path)


    def get_offsets(self, fn):

        # key = line, value = byte
        offsets = {}
        line_count = 0
        with open(fn, 'rb') as f:
            # tell gets end of line, need to get offset before loop for first value
            start = f.tell()
            offsets[line_count] = start
            while f.readline():
                line_count += 1
                offset = f.tell()
                offsets[line_count] = offset

        # discard last offset, blank space at end
        del offsets[line_count]

        return offsets


    def __len__(self):

        return len(self.centers_offsets)

    def __getitem__(self, idx):

        center_idx = self.centers_offsets[idx]
        context_idx = self.contexts_offsets[idx]

        with open(self.centers_path, 'r') as ce, open(self.contexts_path, 'r') as co:
            ce.seek(center_idx)
            co.seek(context_idx)

            centers = ce.readline().strip()
            contexts = co.readline().split()

        # to numpy array and reshape to pair with contexts
        contexts = np.array(contexts, dtype=np.int)
        centers = np.full(len(contexts), centers, dtype=np.int)

        ### keep contexts and negatives separate
        # combine contexts (positives) and negatives into single vector, labels is classifier pos or neg
        # must pad contexts_negatives, will have different lengths due to random sample in each batch
        #contexts_negatives = np.array([contexts + negatives], dtype=np.int).squeeze(0)


        # labels = np.concatenate([np.ones(len(contexts)), np.zeros(len(negatives))]).astype(np.int)

        return centers, contexts

