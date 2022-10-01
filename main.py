import torch
import argparse
from model import Model
from tqdm.auto import tqdm
import pandas as pd



parser = argparse.ArgumentParser(description='BioBERT for label similarity.')

parser.add_argument('source', help='file containing source labels.')
parser.add_argument('target', help='file containing target labels.')
parser.add_argument('--out', dest='out', nargs='?', default='out.csv', help='output file.')

parser.add_argument('--device', dest='device', nargs='?', default='cpu', help='main device to run.')

parser.add_argument('--bs', dest='bs', nargs='?', type=int, default=10, help='the amount of labels to compare in single batch')

args = parser.parse_args()



if __name__ == '__main__':

    device = torch.device(args.device)


    model = Model()
    model.load_state_dict(torch.load('complex_bio.pt'))
    model.eval()
    model.to(device)


    with open(args.source, 'r') as f:
        source = list(map(lambda x: x[:-1], f.readlines()))

    with open(args.target, 'r') as f:
        target = list(map(lambda x: x[:-1], f.readlines()))


    out = []
    with torch.no_grad():
        for a in tqdm(source):
            res = model.sims(a, target, bs=args.bs, device=device)
            out.append([a] + list(res.numpy()))



    data = pd.DataFrame(out, columns=['source'] + target)
    data.to_csv(args.out, index=False)
