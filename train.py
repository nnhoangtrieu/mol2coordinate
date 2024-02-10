import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils import clip_grad_norm_
import numpy as np
import pickle
import re
import rdkit
from rdkit.Chem import rdDistGeom
import model.base
from model.base import Transformer
import utils 
from utils import *
import tqdm 
from tqdm import tqdm
import os 
import argparse

parser = argparse.ArgumentParser(description='Description of your script.')
parser.add_argument('--d_model', type=int, default=256, help='model dimension')
parser.add_argument('--d_ff', type=int, default=512, help='feed forward dimension')
parser.add_argument('--head', type=int, default=8, help='number of attention heads')
parser.add_argument('--layer', type=int, default=1, help='number of layers')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--batch', type=int, default=64, help='batch size')
parser.add_argument('--patience', type=int, default=4, help='patience to stop when val loss is not decreasing')
parser.add_argument('--max_len', type=int, default=30, help='batch size')
parser.add_argument('--save_name', type=str, default="your model name", help='batch size')
arg = parser.parse_args()
print('\n\nArgument Setting: ')
for name, value in arg.__dict__.items() :
    print(f'\t{name}: {value}')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'\nNumber of GPU: {torch.cuda.device_count()}')
rdkit.rdBase.DisableLog('rdApp.*') # Disable rdkit warnings



dataset = MyDataset('data/chembl24_canon_train.pickle', max_len=arg.max_len)
train_set, val_set = random_split(dataset, [0.9, 0.1])
vocab = dataset.vocab
train_loader = DataLoader(train_set, batch_size=arg.batch, shuffle=True)
val_loader = DataLoader(val_set, batch_size=arg.batch)

if not os.path.exists(f'checkpoint/{arg.save_name}') :
    os.makedirs(f'checkpoint/{arg.save_name}')

torch.save(vocab, f'checkpoint/{arg.save_name}/vocab.pt')
torch.save(arg.__dict__, f'checkpoint/{arg.save_name}/setting.pt')
with open(f'checkpoint/{arg.save_name}/setting.txt', 'w') as f :
    f.write(str(arg.__dict__))

print(f'\nNumber of data: {len(train_set)}')
print(f'\nVocab: {vocab}\n')


model = Transformer(arg.d_model, arg.d_ff, arg.head, arg.layer, arg.dropout, vocab).to(device)
loss_fn = nn.L1Loss()
optim = torch.optim.Adam(model.parameters(), lr = arg.lr)


print('\n\n')
print('====================================================')
print('======================TRAINING======================')
print('====================================================')
print('\n\n')


for epoch in range(arg.epoch) : 
    patience = 0 
    best_loss = float('inf')
    train_loss, val_loss = 0, 0
    model.train() 

    for src, x, y, z, tgt in tqdm(train_loader, desc=f'Epoch {epoch+1}') : 
        src, x, y, z, tgt = src.to(device), x.to(device), y.to(device), z.to(device), tgt.to(device)
        src_mask = (src != vocab['<PAD>']).unsqueeze(-2)
        x_mask = get_mask(x.squeeze(-1)[:, :-1])
        y_mask = get_mask(y.squeeze(-1)[:, :-1])
        z_mask = get_mask(z.squeeze(-1)[:, :-1])

        pred = model(src, x[:, :-1], y[:, :-1], z[:, :-1], src_mask, x_mask, y_mask, z_mask)
        loss = loss_fn(pred, tgt[:, 1:, :])
        train_loss += loss.item() 
        loss.backward(), optim.step(), optim.zero_grad(), clip_grad_norm_(model.parameters(), 5)

    model.eval()
    with torch.no_grad() : 
        for src, x, y, z, tgt in val_loader : 
            src, x, y, z, tgt = src.to(device), x.to(device), y.to(device), z.to(device), tgt.to(device)
            src_mask = (src != vocab['<PAD>']).unsqueeze(-2)
            x_mask = get_mask(x.squeeze(-1)[:, :-1])
            y_mask = get_mask(y.squeeze(-1)[:, :-1])
            z_mask = get_mask(z.squeeze(-1)[:, :-1])

            pred = model(src, x[:, :-1], y[:, :-1], z[:, :-1], src_mask, x_mask, y_mask, z_mask)
            loss = loss_fn(pred, tgt[:, 1:, :])
            val_loss += loss.item()



    if val_loss < best_loss :
        patience = 0
        best_loss = val_loss
        torch.save(model.state_dict(), f'checkpoint/{arg.save_name}/{arg.save_name}.pt')
    else : 
        patience += 1
        if patience > arg.patience : 
            print(f'Early Stopping at Epoch {epoch+1}')
            print(f'Best Loss: {best_loss:.3f}')
            print(f'Best model checkpoint is saved at checkpoint/{arg.save_name}/{arg.save_name}.pt')
            break
    
    print(f'\nEpoch {epoch+1} | Train Loss: {train_loss / len(train_loader):.3f} | Val Loss: {val_loss / len(val_loader):2f}\n')
