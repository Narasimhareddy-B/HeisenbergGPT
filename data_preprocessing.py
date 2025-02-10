import collections
import re
from nltk import wordpunct_tokenize, sent_tokenize
import os
from bpe_tokenizer import BPETokenizer
from argparse import ArgumentParser
from multiprocessing import Pool
from itertools import repeat
from tqdm import tqdm, trange
from typing import List, Dict, IO
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s : %(name)s : %(message)s')

file_handler = logging.FileHandler(f'{os.getcwd()}/logs/preprocess.log')
file_handler.setFormatter(formatter)

IOhandler = logging.StreamHandler()
IOhandler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(IOhandler)



class BreakingBadDataset:
    def __init__(self, file_dir: str, out_dir: str) -> None:
        try: 
            self.file_dir = file_dir
            self.out_dir = out_dir
            self.text = None
            self.actors = []
        except Exception as e:
            logger.error(f'error initializing BreakingBadDataset : {e}')


    def read_data(self, file_dir):
        try:

            file_names = os.listdir(file_dir)
            text_final = []
            for file_name in file_names:
                logger.info(f'reading file : {file_name}')
                file_path = os.path.join(file_dir, file_name)
                with open(file_path , 'r', encoding='utf-8-sig') as file:
                    text = file.read()
                    text = text.lower()
                    logger.info(f'{file_name}"s size :{len(text)}')
                    text_final.append(text)
                    logger.info(f'Commulative text size :{len(text_final)}')
            self.text = '\n'.join(text_final)
            logger.info(f'Final text size :{len(self.text)}')
        except Exception as e:
            logger.error(f'error reading data : {e}')



    def clean_data(self):
        try:
            file_names = os.listdir(self.file_dir)
            text_data = []
            for file_name in file_names:
                logger.info('cleaning file : ', file_name)
                file_path = os.path.join(self.file_dir, file_name)
                with open(file_path, 'r', encoding='utf-8-sig') as file:
                    text = file.read()
                    text = text.lower()
                lines = text.split('\n')
                logger.info(f'Number of lines in {file_name} : {len(lines)}')
                new_lines = []
                for line in lines:
                    pattern = r'\[\[(?:[^|]*\|)?([^\]]+)\]\]'
                    #matches = re.findall(pattern, line)
                    def replace_match(match):
                        return match if match in self.actors else ''
                    line = re.sub(pattern, lambda m: replace_match(m.group(1)), line)    
                    #print('\n',line)
                    line = re.sub(r'<.*?>', '', line)
                    line = re.sub(r":'''(\w+)''':", r'\1:', line)
                    line = re.sub(r":'''(\w+):'''", r'\1:', line)
                    new_lines.append(line)
                with open(f'{self.out_dir}/cleaned_{file_name}', 'w', encoding='utf-8-sig') as file:
                    file.write('\n'.join(new_lines))
                text_data.append('\n'.join(new_lines))
                self.text = text_data    
            logger.info(f'Final text size :{len(self.text)}')
        except Exception as e:
            logger.error(f'error cleaning data : {e}')
    
    def get_actornames(self):

        try:

            if self.text is None:
                logger.info(f'\nNo subtitles data in this file : {self.file_path}')
            else:
                lines = self.text.split('\n')
                for line in lines:
                    matches = re.findall(r":'''(\w+)''':", line)
                    for match in matches:
                        if match not in self.actors:
                            self.actors.append(match) 
            return self.actors
        except Exception as e:
            logger.error(f'error getting actor names : {e}')
    
    def get_text(self):
        return self.text
    
    def __str__(self):
        return f'File path : {self.file_dir}'
    
    def __repr__(self):
        return f'BreakingBadDataset({self.file_dir})'
    
    def __len__(self):
        return len(self.text.split('\n'))
    
    def __getitem__(self, idx):
        return self.text.split('\n')[idx]
    
    def __iter__(self):
        self.idx = 0
        return self
    


# class Vocab:
#     def __init__(self, tokens, min_freq=1, special_tokens=[]):
        
#         if tokens and isinstance(tokens[0], list):
#             tokens = [token for sentence in tokens for token in sentence]
#         counter = collections.Counter(tokens)
#         self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
#         self.min_freq = min_freq
#         self.special_tokens = special_tokens
#         self.tokens = list(sorted(set(['<unk>'] + special_tokens + [token for token, freq in self.token_freqs if freq >= min_freq])))
#         print(f'Vocab size : {len(self.tokens)}') 
#         self.token_to_idx = {token : idx for idx, token in enumerate(self.tokens)}
#         self.idx_to_token = {idx : token for idx, token in enumerate(self.tokens)}

#     def __len__(self):
#         return len(self.tokens)
    
#     def __getitem__(self, tokens):
#         if isinstance(tokens, (list, tuple)):
#             return [self.__getitem__(token) for token in tokens]
#         return self.token_to_idx.get(tokens, self.token_to_idx['<unk>'])
    
#     def to_tokens(self, indices):
#         if hasattr(indices, '__len__') and len(indices) > 1:
#             return [self.idx_to_token[int(index)] for index in indices]
#         return self.idx_to_token[int(indices)]
#     def __str__(self):
#         return f'Vocab : {len(self.tokens)} tokens'
    

    
def get_line_ids(line: str, tokenizer: BPETokenizer) -> List[int]:
    
    try:
        tokens = wordpunct_tokenize(line)

        tokens = [list(token) + [tokenizer.get_eow()] for token in tokens]
        lineids = []

        for token in tokens:
            token = tokenizer.merge_bytes(token)
            ids = tokenizer.get_byte_ids(token)
            lineids += ids
        
        sol = tokenizer.get_sol()
        eol = tokenizer.get_eol()
        lineids = [sol] + lineids + [eol]
        return lineids
    except Exception as e:
        logger.error(f'error getting lines ids : {e}')


def tokenize_file(file_path: str, outdir: str, tokenizer: BPETokenizer, line_length: int)-> None:
        
        print(f'tokenizing file path: {file_path}')    
        file_name = file_path[0].split('\\')[-1].split('.')[0]  #file_path.split('/')[-1]
        print(f'tokenizing this file : {file_name}') 
        outpath = f"{outdir}/{file_name}_tokenized.txt"
        lines = sent_tokenize(open(file_path[0], 'r', encoding='utf-8-sig').read())
        
        
        tokens = []
        for line in lines:
            if len(line) > 1: 
                tokens += get_line_ids(line, tokenizer)
        start, end = 0, line_length
        with open(outpath, 'w') as outfile:
            while start < len(tokens):
                if len(tokens[start: end]) == line_length:
                    outstr = ' '.join([str(token) for token in tokens[start: end]])
                    outfile.write(f'{outstr}\n')
                start += line_length
                end += line_length
        logger.info(f'No of tokens in {file_name} : {len(tokens)}')
        logger.info(f'{file_name} tokenized and saved to {outpath}')
        # except Exception as e:
        #     logger.error(f'error tokenizing file: {e}')



def get_metafile(file_dir: str) -> IO:
    try:

        file_names = os.listdir(file_dir)
        file_abs_paths = [os.path.join(file_dir, file_name) for file_name in file_names]
        with open('metafile.txt', 'w') as file:
            for file_name in file_abs_paths:
                file.write(f'{file_name}\n')
        return 'metafile.txt'
    except Exception as e:
        logger.error(f'error getting metafile : {e}')
                 
    
def preprocess_files(inpath: str, outpath: str) -> None: 
    try:

        dataset = BreakingBadDataset(inpath, outpath)
        dataset.read_data(inpath)
        logger.info(f'reading data from {inpath} is completed')

        actors = dataset.get_actornames()
        dataset.clean_data()
        logger.info(f'actors in the dataset : {actors}')
        logger.info('cleaing data is completed')

        metafile = get_metafile(outpath)
        logger.info('metafile is created')
        return metafile
    except Exception as e:
        logger.error(f'error preprocessing files : {e}')


def main():
   
    parser = ArgumentParser()
    parser.add_argument('-c', '--checkpoint', required=True)
    parser.add_argument('-i', '--inpath', required=True)
    parser.add_argument('-o', '--outdir', required=True)
    parser.add_argument('-l', '--line_length', required=True, type=int)
    parser.add_argument('-j', '--jobs', required=True, type=int)
    args = parser.parse_args()

    checkpoint = args.checkpoint
    inpath = args.inpath
    outdir  = args.outdir
    line_length = args.line_length
    jobs = args.jobs

    metafile = preprocess_files(inpath, outdir)

    filepaths = [file.split() for file in open(metafile).readlines()]     
    tokenizer = BPETokenizer.load(checkpoint)

    progress = tqdm(total=len(filepaths), desc='tokenizing files')
    start, end = 0, jobs
    while start < len(filepaths):
        
        paths = filepaths[start: end]

        with Pool(jobs) as pool:
            pool.starmap(
                tokenize_file,
                zip(
                    paths, 
                    repeat(outdir),
                    repeat(tokenizer),
                    repeat(line_length)
                )
            )
        progress.update(jobs)
        start += jobs
        end += jobs
   
if __name__ == '__main__' :

    main()