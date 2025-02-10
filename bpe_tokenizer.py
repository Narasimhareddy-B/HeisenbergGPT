from typing import List, Dict, Tuple
from tqdm import trange, tqdm
import json, re
import os
import logging
from nltk import sent_tokenize, wordpunct_tokenize
from collections import defaultdict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s : %(name)s : %(message)s')

file_handler = logging.FileHandler(f'{os.getcwd()}/logs/tokenizer.log')
file_handler.setFormatter(formatter)

IOhandler = logging.StreamHandler()
IOhandler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(IOhandler)


class BPETokenizer:
    def __init__(self, byte_freqs: Dict[str, int], vocab_to_idx: Dict[str, int], idx_to_vocab: Dict[int, str]):
        self.vocab_to_idx = vocab_to_idx
        self.idx_to_vocab = idx_to_vocab
        self.byte_freqs = byte_freqs
        self.sol = '<line/>'
        self.eol = '</line>'
        self.pad = '<pad>'
        self.unk = '<unk>'
        self.eow = '</w>'
    
    def get_sol(self) -> str:
        return self.sol
    
    def get_eol(self) -> str:
        return self.eol
    
    def get_pad(self) -> str:
        return self.pad
    
    def get_unk(self) -> str:
        return self.unk
    
    def get_eow(self) -> str:
        return self.eow
    
    def get_byte(self, byte_id: int) -> str:
        return self.idx_to_vocab.get(byte_id, self.unk)
    
    def get_byte_id(self, byte: str) -> int:
        unk_id = self.vocab_to_idx[self.unk]
        bid = self.vocab_to_idx[byte] if byte in self.vocab_to_idx else unk_id
        return bid
    
    def merge_bytes(self, bytes_: List[str]) -> List[str]:
        try:
            merged, merged_bytes = self.merge_max_pair(bytes_)
            while merged:
                merged, merged_bytes = self.merge_max_pair(merged_bytes)
            return merged_bytes
        except Exception as e:
            logger.error(f"error merging bytes: {e}")
    
                

    def merge_max_pair(self, bytes_: List[str]) -> List[str]:
        try:
            max_pair = self.get_max_pair_idxs(bytes_)
            if max_pair is None:
                return False, bytes_
            bytes_ = bytes_[:max_pair[0]] + \
                [''.join(bytes_[max_pair[0]: max_pair[1] + 1])]+ \
                    bytes_[max_pair[1]+1:]
            return True, bytes_
        except Exception as e:
            logger.error(f"error merging max pair: {e}")


    def get_max_pair_idxs(self, bytes_: List[str]) -> Tuple[int, int]:
        try:

            pairs = {}  
            for i in range(1, len(bytes_)):
                pair = ''.join(bytes_[i-1: i+1]) 
                if pair in self.byte_freqs:
                    pairs[(i-1, i)] = self.byte_freqs[pair]
            return None if len(pairs) == 0 else max(pairs, key=pairs.get)
        except Exception as e:
            logger.error(f"error getting max pair idxs: {e}")

        
    def get_byte_ids(self, bytes_: List[str]) -> List[int]:
        try:

            ids = []
            for byte in bytes_:
                ids.append(self.vocab_to_idx.get(byte, self.vocab_to_idx[self.unk]))
            return ids
        except Exception as e:
            logger.error(f"error getting byte ids: {e}")


    def get_bytes(self, byte_ids: List[int]) -> List[str]:
        try:

            tokens = []
            for byte_id in byte_ids:
                tokens.append(self.idx_to_vocab.get(byte_id, self.unk))
            return tokens

        except Exception as e:
            logger.error(f'error getting bytes: {e}')
            

    def save(self, path: str) -> None:
        try:
            with open(f'{path}/byte_freqs.json', 'w', encoding='utf-8') as outfile:
                json.dump(self.byte_freqs, outfile, indent=4, ensure_ascii=False)
            logger.info('saving byte_freqs')
            
            with open(f'{path}/vocab_to_idx.json', 'w', encoding='utf-8') as outfile:
                json.dump(self.vocab_to_idx, outfile, indent=4, ensure_ascii=False)
            logger.info('saving vocab to idx in json')

            with open(f'{path}/idx_to_vocab.json', 'w', encoding='utf-8') as outfile:
                json.dump(self.idx_to_vocab, outfile, indent=4, ensure_ascii=False)
            logger.info('saving idx to vocab in json')

        except Exception as e:
            logger.error(f'Error saving tokenizer"s data : {e}')



    @staticmethod
    def load(path: str) -> 'BPETokenizer':
        try:

            logging.info('loading tokenizer')
            with open(f'{path}/byte_freqs.json', 'r', encoding='utf-8', errors="ignore") as infile:
                byte_freqs = json.load(infile)
            logging.info('loading vocab_to_idx')
            with open(f'{path}/vocab_to_idx.json', 'r', encoding='utf-8', errors="ignore") as infile:
                vocab_to_idx = json.load(infile)
            logging.info('loading idx_to_vocab')
            with open(f'{path}/idx_to_vocab.json', encoding='utf-8', errors="ignore") as infile:
                idx_to_vocab = json.load(infile)
            
            return BPETokenizer(byte_freqs, vocab_to_idx, idx_to_vocab)
        
        except Exception as e:
            logging.error(f"error loading tokenizer's data: {e}")
        

        
    @staticmethod
    def train_tokenizer(file_paths: List[str], min_freq: int, merges: int) -> 'BPETokenizer':
        try:

            vocab = create_vocab(file_paths)
            logger.info('vocab is created')
            logger.info(f'Initial vocab size : {len(vocab)}')

            truncate_vocab(vocab, min_freq)
            logger.info(f'vocab is truncated to min_freq: {min_freq}')
            logger.info(f'vocab size after truncation : {len(vocab)}')

            bpe_vocab = create_bpe_vocab(vocab)
            logger.info('bpe vocab is created')
            logger.info(f'BPE vocab_size: {len(bpe_vocab)}')

            for i in trange(merges, desc='Merging Pairs'):
                pairs = get_stats(bpe_vocab)
                if len(pairs) == 0:
                    break
                max_pair = max(pairs, key=pairs.get)
                bpe_vocab = merge_vocab(max_pair, bpe_vocab)
            
            byte_freqs = create_bytes_freq(bpe_vocab)
            logger.info(f'byte_freqs size : {len(byte_freqs)}')
            vocab_to_idx, idx_to_vocab = create_vocab_maps(byte_freqs)
            logger.info(f'vocab to idx size : {len(vocab_to_idx)}')
            return BPETokenizer(byte_freqs, vocab_to_idx, idx_to_vocab)
        except Exception as e:
            logger.error(f"error training tokenizer: {e}")
 

def create_vocab(file_paths:  List[str]) -> Dict[str, int]:
    try:
        vocab = defaultdict(int)
        logging.info('Reading files')
        for file_path in tqdm(file_paths, desc='Reading files'):
            text = open(file_path, 'r', encoding='utf-8-sig').read()
            sentences = sent_tokenize(text.lower())
            for sentence in sentences:
                tokens = wordpunct_tokenize(sentence)
                for token in tokens:
                    vocab[token] +=1
        return vocab
    except Exception as e:
        logging.error(f"error creating vocab : {e}")

def truncate_vocab(vocab: Dict[str, int], min_freq: int) -> None:
    try:

        tokens = list(vocab.keys()) 
        for token in tokens:
            if vocab[token] < min_freq:
                del vocab[token]
    except Exception as e:
        logging.error(f"error truncating vocab : {e}")

def create_bpe_vocab(vocab: Dict[str, int]):
    try:
        bpe_vocab = {}
        for token in vocab:
            new_token = ' '.join(list(token)) + ' </w>'
            bpe_vocab[new_token] = vocab[token]
        return bpe_vocab
    except Exception as e:
        logger.error(f"error creating bpe vocab : {e}")

def get_stats(vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
    try:
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs
    except Exception as e:
        logger.error(f"error getting stats of pairs: {e}")

def merge_vocab(pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str,int]:
    try:

        merged_vocab = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in vocab:
            new_word = p.sub(''.join(pair), word)
            merged_vocab[new_word] = vocab[word]
        return merged_vocab
    except Exception as e:
        logger.error(f"error merging vocab: {e}")

def create_bytes_freq(vocab: Dict[str, int]) -> Dict[str, int]:
    try:
        byte_freq = defaultdict(int)
        for word, freq in vocab.items():
            bytes_ = word.split(' ')
            for byte in bytes_:
                byte_freq[byte] += 1
        
        for token in ['<line/>', '</line>', '<pad>', '<unk>']:
            byte_freq[token] += 1
        return byte_freq
    except Exception as e:
        logger.error(f"error creating bytes freq: {e}")

def create_vocab_maps(vocab: Dict[str, int]) -> (Dict[str, int], Dict[int, str]):
    try:
        ordered_freqs = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        
        vocab_to_idx, idx_to_vocab = {}, {}

        for i in range(len(ordered_freqs)):
            word, freq = ordered_freqs[i]
            vocab_to_idx[word] = i
            idx_to_vocab[i] = word
        return vocab_to_idx, idx_to_vocab
    except Exception as e:
        logger.error(f"error creating vocab maps: {e}")