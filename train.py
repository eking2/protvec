import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import PreprocessVocab, ProtVecVocab, pad_seqs
from model import skip_gram
from tqdm.auto import tqdm
import time
import argparse

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus', type=str, 
                        help='Path to corpus file containing amino acid sequences')
    parser.add_argument('-m', '--min', type=int, default=3,
                        help='Minimum frequency for kmer (default: 3)')
    parser.add_argument('-n', '--context', type=int, default=25,
                        help='Context size (default: 25)')
    parser.add_argument('-k', '--ngram', type=int, default=3,
                        help='Ngram size (default: 3')
    parser.add_argument('-g', '--neg', type=int, default=5,
                        help='Number of negative samples (default: 5)')
    parser.add_argument('-b', '--batch', type=int, default=128,
                        help='Batch size (default: 128')

    parser.add_argument('-t', '--train', action='store_true',
                        help='Train model')
    parser.add_argument('-e', '--embed', type=int, default=100,
                        help='Embedding dimension size (default: 100')
    parser.add_argument('-l', '--lr', type=float, default=0.003,
                        help='Learning rate (default: 0.003')
    parser.add_argument('-p', '--epochs', type=int, default=5,
                        help='Epochs (default: 5)')
    parser.add_argument('-o', '--output', type=str, default='outputs/embeds.txt',
                        help='Embedding vectors output filename')
    
    return parser.parse_args()


def train(model, loader, optimizer):

    model.train()
    losses = 0

    for batch_idx, (centers, contexts_negatives, labels, masks) in enumerate(loader):

        centers = centers.to(device).long()
        contexts_negatives = contexts_negatives.to(device).long()
        labels = labels.to(device)
        masks = masks.to(device)

        # every batch needs a new criterion, different padding
        criterion = nn.BCEWithLogitsLoss(weight=masks, reduction='none')

        pred = model(centers, contexts_negatives)
        loss = criterion(pred.view_as(labels), labels.float())

        # normalize due to different length
        loss = (loss.mean(dim=1) / masks.sum(dim=1) * masks.shape[1]).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += loss.item()

    return losses / len(loader)


def save_embeds(model, dataset, embed_size, filename):

    vocab_size = len(dataset.word2idx)

    with open(filename, 'w') as fo:
        fo.write(f'{vocab_size} {embed_size}\n')

        for word_idx, word in dataset.idx2word.items():

            word_idx = torch.tensor(int(word_idx)).to(device).long()
            embed = model.in_emb(word_idx)

            # write out word embedding values
            embed = ' '.join(map(str, embed.tolist()))
            fo.write(f'{word} {embed}\n')


if __name__ == '__main__':

    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # full uniprot dataset takes ~1hr
    if args.corpus:
        PreprocessVocab(args.corpus, k=args.ngram, min_freq=args.min, window=args.context,
                        neg_samples=args.neg).run_all()

    if args.train:

        dataset = ProtVecVocab()
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, collate_fn=pad_seqs)

        # centers, contexts_negatives, labels, mask = next(iter(loader))
        vocab_size = len(dataset.word2idx)

        # print('centers')
        # print(centers.shape)
        # print('contexts_negatives')
        # print(contexts_negatives.shape)
        # print('labels')
        # print(labels.shape)
        # print('mask')
        # print(mask.shape)
        
        model = skip_gram(vocab_size, args.embed).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        loss = 0
        pbar = tqdm(range(args.epochs))
        for epoch in pbar:

            pbar.set_postfix({'Epoch' : epoch,
                              'Loss' : loss})

            loss = train(model, loader, optimizer)
            #print(f'Epoch: {epoch+1:3} | Loss: {loss:.2f}')

        save_embeds(model, dataset, args.embed, args.output)
