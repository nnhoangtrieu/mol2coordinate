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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rdkit.rdBase.DisableLog('rdApp.*') # Disable rdkit warnings

dataset = MyDataset('data/chembl24_canon_train.pickle', max_len=arg.max_len)
train_set, val_set = random_split(dataset, [0.9, 0.1])
vocab = dataset.vocab

train_loader = DataLoader(train_set, batch_size=arg.batch, shuffle=True)
val_loader = DataLoader(val_set, batch_size=arg.batch)

model = Transformer(arg.d_model, arg.d_ff, arg.head, arg.layer, arg.dropout, vocab).to(device)
loss_fn = nn.L1Loss()
optim = torch.optim.Adam(model.parameters(), lr = arg.lr)



for epoch in range(arg.epoch) : 
    train_loss, val_loss = 0, 0
    model.train() 

    for src, x, y, z, tgt in train_loader : 
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
    print(f'Epoch {epoch+1} | Train Loss: {train_loss / len(train_loader)} | Val Loss: {val_loss / len(val_loader)}')
