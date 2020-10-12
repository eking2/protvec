from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def load_embeddings(embeddings):

    df = pd.read_csv(embeddings, delim_whitespace=True, skiprows=1, header=None)
    #embeddings = df.iloc[:, 1:].values
    #kmers = df.iloc[:-1, 0]  # do not include <unk>

    #return kmers, embeddings
    return df.iloc[:-1, :]  # do not include <unk>


def tsne_plots(output='./outputs/tsne_plots.png'):

    # df columns names to plot titles
    names = {'mass' : 'Mass',
             'charge' : 'Charge',
             'vol' : 'Volume',
             'vdw_vol' : 'Van der Waals Volume',
             'hydrophobicity' : 'Hydrophobicity',
             'polarity' : 'Polarity'}

    # load props
    prop_df = pd.read_csv('./inputs/kmers_props_calc.csv')
    props = ['mass', 'vol', 'vdw_vol', 'polarity', 'hydrophobicity', 'charge']  # manually set order to plot

    # columns to separate embedding data from properties and kmer
    no_embed = prop_df.columns.values.tolist() + [0]

    # load embeddings
    embeddings = load_embeddings('./outputs/embeds.txt')

    # merge
    df = embeddings.merge(prop_df, left_on=0, right_on='kmer')

    # keep only embedding data for TSNE
    embeds = df[df.columns.difference(no_embed)].values
    #embeds_sc = StandardScaler().fit_transform(embeds)

    tsne = TSNE(n_components=2, n_jobs=-1)
    embeds_tsne = tsne.fit_transform(embeds)
    #pca = PCA(n_components=2)
    #embeds_pca = pca.fit_transform(embeds)

    # plot
    fig = plt.figure(figsize=(12, 7))

    # loop each prop separately
    # plt to set each subplots colorbar separately
    for i, prop in enumerate(props):
        plt.subplot(2, 3, i+1)
        p = plt.scatter(embeds_tsne[:, 0], embeds_tsne[:, 1], c=df[prop], alpha=0.2, s=3)
        cbar = plt.colorbar(p)
        cbar.solids.set(alpha = 1)  # colorbar no transparency
        plt.title(names[prop])

    plt.tight_layout()
    plt.savefig(output, bbox_inches='tight', dpi=300)


if __name__ == '__main__':

    tsne_plots()
