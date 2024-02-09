import rdkit
import torch 
import utils 
from utils import * 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def inference(model, smi, vocab, max_token_len) : 
    num_atom = rdkit.Chem.MolFromSmiles(smi).GetNumAtoms()
    src = get_inference_input(smi, vocab, max_token_len)
    model.eval()
    src = src.to(device)
    src_mask = (src != vocab['<PAD>']).unsqueeze(-2)
    x = torch.zeros(1, 1, 1).to(device) 
    y = torch.zeros(1, 1, 1).to(device)
    z = torch.zeros(1, 1, 1).to(device)

    out = torch.zeros(1, 1, 3).to(device)
    
    for _ in range(num_atom)  :
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

    return out