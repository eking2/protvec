from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE


def load_embeddings(embeddings):

    df = pd.read_csv(embeddings, delim_whitespace=True, skiprows=1, header=None)
    embeddings = df.iloc[:, 1:].values
    kmers = df.iloc[:-1, 0]  # do not include <unk>

    return kmers, embeddings




kmers, embeddings = load_embeddings('./outputs/embeds.txt')
# print(embeddings.shape)
# print(kmers)
calc_props(kmers)
