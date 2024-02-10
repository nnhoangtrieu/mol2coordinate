import rdkit
import torch 
import utils 
from utils import * 
import model.base
from model.base import Transformer
import argparse

parser = argparse.ArgumentParser(description='Description of your script.')
parser.add_argument('--save_name', type=str, default="model", help='save name of your trained model')
parser.add_argument('--input', type=str, default="CCCC", help='model dimension')

arg = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab = torch.load(f'checkpoint/{arg.save_name}/vocab.pt')
setting = torch.load(f'checkpoint/{arg.save_name}/setting.pt')
print('The maximumm length of SMILES for this model is: ', setting['max_len'])

if arg.input > setting['max_len'] :
    print(f'Your input is longer than the maximum length of SMILES. Please input SMILES with length less than {setting["max_len"]}')
    exit()

model = Transformer(setting['d_model'], setting['d_ff'], setting['head'], setting['layer'], setting['dropout'], vocab).to(device)
model.load_state_dict(torch.load(f'checkpoint/{arg.save_name}/model.pt'))

model.eval()
pred = inference(model, arg.input, vocab, setting['max_len'], True)


print(f'Coordinate: {pred.data}')

