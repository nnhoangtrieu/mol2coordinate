import os

smi = "CCC"
if not os.path.exists(f'attention-inference/{smi}') : 
    os.makedirs(f'attention-inference/{smi}/cross-attn')
    os.makedirs(f'attention-inference/{smi}/self-attn')