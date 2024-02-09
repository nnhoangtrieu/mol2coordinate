import torch 
from torch.utils.data import Dataset 
import re 
import pickle
import rdkit
import numpy as np 
import multiprocessing
import argparse

parser = argparse.ArgumentParser(description='Description of your script.')

parser.add_argument('--d_model', type=int, default=256, help='model dimension')
parser.add_argument('--d_ff', type=int, default=512, help='feed forward dimension')
parser.add_argument('--head', type=int, default=8, help='number of attention heads')
parser.add_argument('--layer', type=int, default=4, help='number of layers')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs')
parser.add_argument('--batch', type=int, default=128, help='batch size')
parser.add_argument('--max_len', type=int, default=128, help='batch size')
parser.add_argument('--save_name', type=int, default=128, help='batch size')


arg = parser.parse_args()
def parallel_f(f, input_list) :
    pool = multiprocessing.Pool()
    return pool.map(f, input_list)

def parallel_f_with_args(f, input_list, args) :
    pool = multiprocessing.Pool()
    return pool.starmap(f, zip(input_list, args))

def subsequent_mask( size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )

    return subsequent_mask == 0

def get_mask(target, pad_value = -1) :
    mask = (target != torch.tensor(pad_value)).unsqueeze(-2)
    return mask & subsequent_mask(target.size(-1)).type_as(mask.data)

class MyDataset(Dataset):
    def __init__(self, path, max_len):
        self.path = path
        self.max_len = max_len
        self.token_list, self.coor_list = self.process()

    def extract(self):
        if self.path.lower().endswith('.txt'):
            with open(self.path, 'r') as file:
                data = [line.strip() for line in file if len(line) < self.max_len]
            return data
        elif self.path.lower().endswith('.pickle'):
            with open(self.path, 'rb') as file:
                data = pickle.load(file)
                data = [x for x in data if len(x) < self.max_len]
            return data
        else:
            raise ValueError("Unsupported file format. Only .txt and .pkl files are supported.")
    
    def tokenizer(self, smile):
        pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regezz = re.compile(pattern)
        tokens = [token for token in regezz.findall(smile)]
        assert smile == ''.join(tokens), ("{} could not be joined".format(smile))
        return tokens
    
    def get_vocab(self, smi_list) :
        vocab = {'<START>': 0, '<END>': 1, '<PAD>': 2}
        for smi in smi_list :
            for char in smi :
                if char not in vocab :
                    vocab[char] = len(vocab) 
        return vocab 
    
    def get_coor(self, smi) :
        mol = rdkit.Chem.MolFromSmiles(smi)
        if mol is None : 
            print(smi)
        num_atom = mol.GetNumAtoms()
        try : 
            mol_h = rdkit.Chem.AddHs(mol)
            rdkit.Chem.rdDistGeom.EmbedMolecule(mol_h)
            conformer = mol_h.GetConformer()
            coor = conformer.GetPositions()
        except :
            print('Error in embedding molecule and will be removed: ', smi)
            return None
        coor = np.round(coor, 2)
        x, y, z = coor[0] 
        end_coor = np.ones((1, 3)) 
        normalize_coor = coor[:num_atom] - [x, y, z]

        return np.concatenate((normalize_coor, end_coor), axis=0)
    
    def encode(self, smi, vocab) :
        return [vocab['<START>']] + [vocab[char] for char in smi] + [vocab['<END>']]
    
    def pad_smi(self, smi, max_len) :
        return smi + [2] * (max_len - len(smi))
    
    def pad_coor(self, coor, max_len) : 
        pad = np.ones((max_len - coor.shape[0], 3)) * -1
        return np.concatenate((coor, pad), axis=0)
        
    def process(self) : 
        smi_list = self.extract() 

        coor_list = parallel_f(self.get_coor, smi_list)
        smi_list = [s for i, s in enumerate(smi_list) if coor_list[i] is not None]
        coor_list = [c for c in coor_list if c is not None]
        max_coor_len = max(coor_list, key=lambda x : x.shape[0]).shape[0]
        coor_list = [self.pad_coor(c, max_coor_len) for c in coor_list]

        smi_list = parallel_f(self.tokenizer, smi_list)

        self.vocab = self.get_vocab(smi_list) 
        inv_vocab = {v:k for k, v in self.vocab.items()}

        token_list = [self.encode(s, self.vocab) for s in smi_list]

        max_token_len = len(max(token_list, key=len))
        token_list = [self.pad_smi(t, max_token_len) for t in token_list]

        return token_list, coor_list

    def __len__(self) : 
        return len(self.token_list) 
    
    def __getitem__(self, idx) : 
        src = torch.tensor(self.token_list[idx], dtype=torch.long)
        tgt = torch.tensor(self.coor_list[idx], dtype=torch.float)
        x, y, z = torch.chunk(tgt, chunks=3 ,dim =1)
        return src, x, y, z, tgt
