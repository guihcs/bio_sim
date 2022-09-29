import torch

from model import Model
from tqdm.auto import tqdm
import pandas as pd
import sys

if __name__ == '__main__':
    if len(sys.argv) > 3:
        device = torch.device(sys.argv[-1])
    else:
        device = torch.device('cpu')

    model = Model()
    model.load_state_dict(torch.load('complex_bio.pt'))
    model.eval()
    model.to(device)


    with open(sys.argv[1], 'r') as f:
        source = list(map(lambda x: x[:-1], f.readlines()))

    with open(sys.argv[2], 'r') as f:
        target = list(map(lambda x: x[:-1], f.readlines()))


    out = []

    for a in tqdm(source):
        res = model.sims(a, target, device=device)
        out.append([a] + list(res.numpy()))



    data = pd.DataFrame(out, columns=['source'] + target)
    data.to_csv(sys.argv[3], index=False)
