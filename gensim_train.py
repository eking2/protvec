import gensim
from dataset import PreprocessVocab
from gensim.models import Word2Vec, callbacks
import multiprocessing as mp
import argparse
import logging

# log to file and console
logging.basicConfig(filename='outputs/gensim_train.log',
                    format='%(asctime)s : %(levelname)s : %(message)s', 
                    level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())

# train using gensim
# manual implementation is too slow/memory intensive

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus', type=str)
    parser.add_argument('-k', '--ngram', type=int, default=3)
    parser.add_argument('-m', '--min', type=int, default=3)
    parser.add_argument('-n', '--context', type=int, default=25)
    parser.add_argument('-g', '--neg', type=int, default=5)
    parser.add_argument('-e', '--embed', type=int, default=100)
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-p', '--epochs', type=int, default=5)

    return parser.parse_args()


def tokenize(args):

    tokenizer = PreprocessVocab(args.corpus, args.ngram, args.min, args.context)
    tokenizer.gensim_setup()


def save_embeddings(model, args):

    with open('outputs/embeds.txt', 'w') as fo:
        fo.write(f'{len(model.wv.vocab)} {args.embed}\n')

        for i, word in enumerate(model.wv.vocab):
            embeds = ' '.join(map(str, model.wv[word]))
            fo.write(f'{word} {embeds}\n')


class callback(callbacks.CallbackAny2Vec):

    '''print epoch loss at end of epoch'''

    def __init__(self):
        self.epoch = 0
        self.last_loss = 0

    def on_epoch_end(self, model):

        # accumulates total loss over all epochs, not per epoch
        loss = model.get_latest_training_loss()
        epoch_loss = loss - self.last_loss
        self.last_loss = loss
        logging.info(f'Epoch: {self.epoch + 1:02} | Loss: {epoch_loss:,.2f}')
        self.epoch += 1


def run_word2vec(args):

    n_cpus = mp.cpu_count() - 1

    # input linesentence format
    window = (args.context // 2) + 1
    model = Word2Vec(corpus_file='./preprocessed/corpus.txt', size=args.embed, window=args.context,
                    min_count=args.min, negative=args.neg, workers=n_cpus, iter=args.epochs,
                    sorted_vocab=1, compute_loss = True, callbacks=[callback()], sg=1)

    # save embeddings
    model.save('outputs/protvec.mdl')
    save_embeddings(model, args)

if __name__ == '__main__':

    args = parse_args()

    if args.corpus:
        tokenize(args)

    if args.train:
        run_word2vec(args)
