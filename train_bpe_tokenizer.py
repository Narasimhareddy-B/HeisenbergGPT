from  bpe_tokenizer import BPETokenizer
import os
import logging 
from argparse import ArgumentParser

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s : %(name)s : %(message)s')

file_handler = logging.FileHandler(f'{os.getcwd()}/logs/train_tokenizer.log')
file_handler.setFormatter(formatter)

IOhandler = logging.StreamHandler()
IOhandler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(IOhandler)

def main():
    try:

        parser = ArgumentParser()
        parser.add_argument('-i', '--inpath', required=True)
        parser.add_argument('-o', '--outpath', required=True)
        parser.add_argument('-m', '--merges', required=True, type=int)
        parser.add_argument('-n', '--mincount', required=True, type=int)
        args = parser.parse_args()
        outpath = args.outpath
        inpath = args.inpath
        merges = args.merges
        mincount = args.mincount

        wiki_file = [f'{inpath}/{os.listdir(inpath)[0]}']
        print(f'wiki_file: {wiki_file}')
        bpe_tokenizer = BPETokenizer.train_tokenizer(wiki_file, mincount, merges)
        logger.info('training tokenizer is completed')
        bpe_tokenizer.save(outpath)
        logger.info('tokenizer is saved')
    except Exception as e:
        logger.error(f"error training tokenizer: {e}")



if __name__ == '__main__':
    main()
