import torch 
from torch.utils.data import Dataset 
import re 
import pickle
import rdkit
import numpy as np 




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
        mol_h = rdkit.Chem.AddHs(mol)
        rdkit.Chem.rdDistGeom.EmbedMolecule(mol_h)
        conformer = mol_h.GetConformer()
        coor = conformer.GetPositions()
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

        coor_list = [self.get_coor(s) for s in smi_list]
        max_coor_len = max(coor_list, key=lambda x : x.shape[0]).shape[0]
        coor_list = [self.pad_coor(c, max_coor_len) for c in coor_list]

        smi_list = [self.tokenizer(s) for s in smi_list]

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
