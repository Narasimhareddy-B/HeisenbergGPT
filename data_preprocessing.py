import collections
import re
import nltk
from nltk import wordpunct_tokenize, sent_tokenize
import os


class BreakingBadDataset:
    def __init__(self, file_dir):
        self.file_dir = file_dir
        self.text = None
        self.actors = []
    
    def read_data(self, file_dir):
        file_names = os.listdir(file_dir)
        text_final = []
        for file_name in file_names:
            print('reading file : ', file_name)
            file_path = os.path.join(file_dir, file_name)
            with open(file_path , 'r', encoding='utf-8-sig') as file:
                text = file.read()
                text = text.lower()
                print(f'{file_name}"s size :{len(text)}')
                text_final.append(text)
                print(f'Commulative text size :{len(text_final)}')
        self.text = '\n'.join(text_final)
        print(f'Final text size :{len(self.text)}')

    def clean_data(self):
        lines = self.text.split('\n')
        print(f'Number of lines in the text : {len(lines)}')
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
        self.text = '\n'.join(new_lines)      

    def tokenize(self):
        sentences = sent_tokenize(self.text)
        tokens = [token for sentence in sentences for token in wordpunct_tokenize(sentence)]
        return tokens
 
    def get_actornames(self):

        if self.text is None:
            print(f'\nNo subtitles data in this file : {self.file_path}')
        else:
            lines = self.text.split('\n')
            for line in lines:
                matches = re.findall(r":'''(\w+)''':", line)
                for match in matches:
                    if match not in self.actors:
                        self.actors.append(match) 
        return self.actors
    
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
    

class Vocab:
    def __init__(self, tokens, min_freq=1, special_tokens=[]):
        
        if tokens and isinstance(tokens[0], list):
            tokens = [token for sentence in tokens for token in sentence]
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.min_freq = min_freq
        self.special_tokens = special_tokens
        self.word2idx = {}
        self.idx2word = {}
        
       