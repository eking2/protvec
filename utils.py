import numpy as np
from pathlib import Path

AA = 'ACDEFGHIKLMNPQRSTVWY'

def seq_to_kmers(seq, k=3):

    '''chunk sequence into sliding k-mers

    Parameters
    ----------
    seq (str) : amino acid sequence
    k (int) : k-mer length

    Returns
    -------
    kmers (list k-mers) : each sequence split into k-mers, separate list for each reading frame 
    '''

    kmers = []
    for i in range(k):
        # each reading frame 
        slide = seq[i:]
        frame = [slide[i:i+k] for i in range(0, len(seq), k) if len(slide[i:i+k]) == k]
        kmers.append(frame)

    return kmers


def gen_corpus(fn, k):

    '''generate corpus of k-mers from input sequence file

    Parameters
    ----------
    fn (str) : path to sequence file
    k (int) : k-mer length

    Returns
    -------
    corpus (list) : list of k-mers
    '''

    lines = Path(fn).read_text().splitlines()

    corpus = []
    for line in lines:
        # sequence lines capitalized, comments lower
        if line[0].isupper():
            kmers = seq_to_kmers(line)
            corpus.extend(kmers)

    return corpus


def aa_to_ohe(aa_seq):

    '''one hot encode amino acid sequence

    Parameters
    ----------
    aa_seq (str) : sequence of amino acids

    Returns
    -------
    aa_ohe (np array) : one hot encoding of aa_seq
    '''

    labels = [AA.index(aa) for aa in aa_seq]
    aa_ohe = np.eye(len(AA))[labels]

    return aa_ohe


def ohe_to_aa(ohe):

    '''convert one hot encoding to amino acid sequence'''

    labels = np.argmax(ohe, axis=1)
    aa_seq = ''.join([AA[i] for i in labels])

    return aa_seq


