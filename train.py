import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import PreprocessVocab, ProtVecVocab, pad_seqs
from model import skip_gram
import time
import argparse

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus', type=str, 
                        help='Path to corpus file containing amino acid sequences')
    parser.add_argument('-b', '--batch', type=int, default=128,
                        help='Batch size (default: 128')
    parser.add_argument('-e', '--embed', type=int, default=300,
                        help='Embedding dimension size (default: 300')
    parser.add_argument('-l', '--lr', type=float, default=0.003,
                        help='Learning rate (default: 0.003')
    parser.add_argument('-p', '--epochs', type=int, default=5,
                        help='Epochs (default: 5)')
    
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
        criterion = nn.BCEWithLogitsLoss(weight=masks)

        pred = model(centers, contexts_negatives)
        loss = criterion(pred.view_as(labels), labels.float())

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

            # to list to save  
            embed = ' '.join(map(str, embed.tolist()))
            fo.write(f'{word} {embed}\n')


if __name__ == '__main__':

    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # full uniprot dataset takes ~1hr
    if args.corpus:
        PreprocessVocab(args.corpus).run_all()

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

    # losses = []
    # for epoch in range(args.epochs):

    #     loss = train(model, loader, optimizer)
    #     losses.append(loss)

    # print(losses)
    save_embeds(model, dataset, args.embed, 'embeds.txt')
