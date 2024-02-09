import torch 
from torch.utils.data import Dataset 
import re 
import pickle
import rdkit
import numpy as np 
import multiprocessing
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# parser = argparse.ArgumentParser(description='Description of your script.')

# parser.add_argument('--d_model', type=int, default=256, help='model dimension')
# parser.add_argument('--d_ff', type=int, default=512, help='feed forward dimension')
# parser.add_argument('--head', type=int, default=8, help='number of attention heads')
# parser.add_argument('--layer', type=int, default=4, help='number of layers')
# parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
# parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
# parser.add_argument('--epoch', type=int, default=100, help='number of epochs')
# parser.add_argument('--batch', type=int, default=128, help='batch size')
# parser.add_argument('--max_len', type=int, default=128, help='batch size')
# parser.add_argument('--save_name', type=int, default=128, help='batch size')


# arg = parser.parse_args()


def visualize_molecule(xyz, smi) : 
    mol = rdkit.Chem.MolFromSmiles(smi)
    print(mol.GetNumAtoms())
    conf = rdkit.Chem.Conformer(mol.GetNumAtoms())
    for i, coor in enumerate(xyz) : 
        conf.SetAtomPosition(i, (coor[0], coor[1], coor[2]))
    mol.AddConformer(conf)
    rdkit.Chem.Draw.MolToMPL(mol)

def encode(smi, vocab) :
    return [vocab['<START>']] + [vocab[char] for char in smi] + [vocab['<END>']]

def pad_smi(smi, max_len) :
    return smi + [2] * (max_len - len(smi))

def tokenizer(smile):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regezz = re.compile(pattern)
    tokens = [token for token in regezz.findall(smile)]
    assert smile == ''.join(tokens), ("{} could not be joined".format(smile))
    return tokens

def get_inference_input(smi, vocab, max_len) :
    smi = tokenizer(smi)
    smi = encode(smi, vocab)
    smi = pad_smi(smi, max_len)
    return torch.tensor(smi, dtype=torch.long).unsqueeze(0)





            

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


def inference(model, smi, vocab, max_token_len, plot = False) : 
    num_atom = rdkit.Chem.MolFromSmiles(smi).GetNumAtoms()
    src = get_inference_input(smi, vocab, max_token_len) if type(smi) == str else smi
    model.eval()

    src = src.to(device)
    src_mask = (src != vocab['<PAD>']).unsqueeze(-2)
    x = torch.zeros(1, 1, 1).to(device) 
    y = torch.zeros(1, 1, 1).to(device)
    z = torch.zeros(1, 1, 1).to(device)
    out = torch.zeros(1, 1, 3).to(device)
    
    for _ in range(num_atom - 1)  :
        x_mask = get_mask(x.squeeze(-1))
        y_mask = get_mask(y.squeeze(-1))
        z_mask = get_mask(z.squeeze(-1))
        pred = model(src, x, y, z, src_mask, x_mask, y_mask, z_mask)
        x_pred, y_pred, z_pred = torch.chunk(pred[:, -1, :], 3, dim = -1)
        
        x = torch.cat((x, x_pred.unsqueeze(1)), dim = 1)
        y = torch.cat((y, y_pred.unsqueeze(1)), dim = 1)
        z = torch.cat((z, z_pred.unsqueeze(1)), dim = 1)

        xyz = torch.cat((x_pred, y_pred, z_pred), dim = -1).unsqueeze(1)
        out = torch.cat((out, xyz), dim = 1)

        self_attn = model.encoder.layers[-1].self_attn.attn
        x_attn = model.xdecoder.layers[-1].src_attn.attn
        y_attn = model.ydecoder.layers[-1].src_attn.attn
        z_attn = model.zdecoder.layers[-1].src_attn.attn
    
    if plot : 
        visualize_attn(x_attn, smi, 'x-cross-attn')
        visualize_attn(y_attn, smi, 'y-cross-attn')
        visualize_attn(z_attn, smi, 'z-cross-attn')
        visualize_attn(self_attn, smi, 'self-attn')
    return out
            
def visualize_attn(attn, smi, type) : 
    token_len = len(tokenizer(smi))
    num_atom = attn.size(-2)
    if not os.path.exists(f'attention-inference/{smi}') : 
        os.makedirs(f'attention-inference/{smi}/x-cross-attn')
        os.makedirs(f'attention-inference/{smi}/y-cross-attn')
        os.makedirs(f'attention-inference/{smi}/z-cross-attn')
        os.makedirs(f'attention-inference/{smi}/self-attn')

    attn = attn.squeeze(0)
    if type == 'self-attn' : 
        attn = attn[:, :token_len, :token_len]
    else :
        attn = attn[:, :, :token_len]
    attn = attn.cpu().detach().tolist() 



    figsize = (15, 8) if type == 'self-attn' else (20, 5)
    fig, axs = plt.subplots(2, 4, figsize=figsize)
    axs = axs.flatten()
    for i, mat in enumerate(attn) : 
        axs[i].xaxis.set_ticks_position('top')
        axs[i].imshow(mat, cmap = 'viridis')
        axs[i].set_xticks(np.arange(token_len))
        axs[i].set_yticks(np.arange(token_len)) if type == 'self-attn' else axs[i].set_yticks(np.arange(num_atom))
        axs[i].set_xticklabels(tokenizer(smi))
        axs[i].set_yticklabels(tokenizer(smi)) if type == 'self-attn' else axs[i].set_yticklabels(np.arange(num_atom))

    plt.tight_layout()
    fig.patch.set_facecolor('white')
    plt.savefig(f'attention-inference/{smi}/{type}/{datetime.datetime.now()}.png')
    

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
        self.max_coor_len = max(coor_list, key=lambda x : x.shape[0]).shape[0]
        coor_list = [self.pad_coor(c, self.max_coor_len) for c in coor_list]

        smi_list = parallel_f(self.tokenizer, smi_list)

        self.vocab = self.get_vocab(smi_list) 
        self.inv_vocab = {v:k for k, v in self.vocab.items()}

        token_list = [self.encode(s, self.vocab) for s in smi_list]

        self.max_token_len = len(max(token_list, key=len))
        token_list = [self.pad_smi(t, self.max_token_len) for t in token_list]

        return token_list, coor_list

    def __len__(self) : 
        return len(self.token_list) 
    
    def __getitem__(self, idx) : 
        src = torch.tensor(self.token_list[idx], dtype=torch.long)
        tgt = torch.tensor(self.coor_list[idx], dtype=torch.float)
        x, y, z = torch.chunk(tgt, chunks=3 ,dim =1)
        return src, x, y, z, tgt
