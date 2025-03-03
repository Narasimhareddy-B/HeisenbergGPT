from random import randint, sample
import torch
from torch import FloatTensor, LongTensor, Tensor, stack, cat
from torch.nn.functional import one_hot
from torch.utils.data import IterableDataset

class TokenIDDataset(IterableDataset):

    def __init__(self, datapath: str, window_size: int, vocab_size: int, 
                 unk: int):
        
        super().__init__()
        self.data = open(datapath, 'r').readlines()
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.unk_token = unk


    def __iter__(self):
        for line_idx in range(len(self.data)):

            line = self.data[line_idx].strip().split()
            start = randint(0, len(line)-self.window_size-1)
            end = start + self.window_size + 1

            ids = LongTensor([int(x) for x in line[start:end]])
            ignore = (ids == self.unk)

            yield ids[:-1], ids[1:], ignore[:-1]
            
    def __len__(self):
        return len(self.data)
    

    @staticmethod
    def collate(batch: Tensor) -> (Tensor, Tensor, Tensor):
        
        xids = cat([batch[i][0][None, :] for i in range(len(batch))], dim=0)
        yids = cat([batch[i][1][None, :] for i in range(len(batch))], dim=0)
        ignore = cat([batch[i][2][None, :] for i in range(len(batch))], dim=0)
        return xids, yids, ignore
    
class TokenIDSubset(TokenIDDataset):
    
    def __init__(self, dataset: TokenIDDataset, size: int):

        self.data = sample(dataset.data, size)
        self.window_size = dataset.window_size
        self.vocab_size = dataset.vocab_size
        self.unk_token = dataset.unk_token

    def __iter__(self):
        yield from super().__iter__()


    def __len__(self):
        return super().__len__()


