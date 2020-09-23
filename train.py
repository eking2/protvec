import torch
import argparse
from torch.utils.data import DataLoader
from dataset import PreprocessVocab, ProtVecVocab, pad_seqs
import time

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus', type=str, 
                        help='Path to corpus file containing amino acid sequences')
    
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    # full uniprot dataset takes ~1hr
    if args.corpus:
        PreprocessVocab(args.corpus).run_all()

    dataset = ProtVecVocab()
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=pad_seqs)

    centers, contexts_negatives, labels, mask = next(iter(loader))

    print('centers')
    print(centers.shape)
    print('contexts_negatives')
    print(contexts_negatives.shape)
    print('labels')
    print(labels.shape)
    print('mask')
    print(mask.shape)
    
