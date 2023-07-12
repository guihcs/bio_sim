# %%
import argparse
import pandas as pd

from transformers import AutoTokenizer, AutoModel

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim


def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='LD similarity.')

    arg_parser.add_argument('source_data')
    arg_parser.add_argument('--output', dest='output', default='./output', help='Folder to save the results.')

    arg_parser.add_argument('--model', dest='model', nargs='?', default='sentence-transformers/all-MiniLM-L6-v2',
                            help='Model used to generate embeddings. Any model from huggingface transformers is expected to work.')
    arg_parser.add_argument('--batch_size', dest='batch_size', nargs='?', type=int, default=32,
                            help='The batch size for the embeddings generation.')
    arg_parser.add_argument('--epochs', dest='epochs', nargs='?', type=int, default=10)

    arg_parser.add_argument('--lr', dest='lr', nargs='?', type=float, default=0.003)

    arg_parser.add_argument('--device', dest='device', nargs='?', default='cpu',
                            help='The device to use for the embeddings generation.')

    return arg_parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()

    output = args.output
    path = args.source_data
    model_name = path.split('/')[-1].split('.')[0]
    data = pd.read_csv(path, header=None, sep='\t')

    data.head()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    src = data[0].tolist()
    trg = data[1].tolist()
    src_enc = tokenizer(src, padding=True, return_tensors='pt')
    trg_enc = tokenizer(trg, padding=True, return_tensors='pt')

    src_ids = src_enc['input_ids']
    src_mask = src_enc['attention_mask']

    trg_ids = trg_enc['input_ids']
    trg_mask = trg_enc['attention_mask']

    dataset = TensorDataset(src_ids, src_mask, trg_ids, trg_mask)

    def emb(model, src_ids, src_mask):
        output = model(src_ids, attention_mask=src_mask, output_hidden_states=True)
        em = output['hidden_states'][-1] * src_mask.unsqueeze(-1)
        em = em.sum(1) / src_mask.sum(-1).unsqueeze(-1)
        return em

    device = torch.device(args.device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.TripletMarginLoss()
    lh = []
    for e in range(args.epochs):
        el = []
        for sid, sm, tid, tm in DataLoader(dataset, batch_size=args.batch_size, shuffle=True):
            optimizer.zero_grad()
            sid = sid.to(device)
            sm = sm.to(device)
            tid = tid.to(device)
            tm = tm.to(device)
            anchor = emb(model, sid, sm)
            positive = emb(model, tid, tm)
            rdi = torch.randint(0, trg_ids.shape[0], (tid.shape[0],))
            negative = emb(model, trg_ids[rdi].to(device), trg_mask[rdi].to(device))
            loss = crit(anchor, positive, negative)
            el.append(loss.item())
            loss.backward()
            optimizer.step()

        lh.append(sum(el) / len(el))

    print(model_name, lh[0], lh[-1])
    # %%
    torch.save(model.state_dict(), f'{output}/{model_name}.pt')
