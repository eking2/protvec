import torch
import argparse
from torch.utils.data import DataLoader
from dataset import Vocab, pad_seqs

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus', type=str, required=True,
                        help='Path to corpus file containing amino acid sequences')
    
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    data = Vocab(args.corpus)
    loader = DataLoader(data, batch_size=32, shuffle=True, collate_fn=pad_seqs)

    centers, contexts_negatives, labels, mask = next(iter(loader))

    print('centers')
    print(centers.shape)
    print('contexts_negatives')
    print(contexts_negatives.shape)
    print('labels')
    print(labels.shape)
    print('mask')
    print(mask.shape)
    
