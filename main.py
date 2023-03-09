import os

import argparse

import pandas as pd
from ldp import parser

from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL
from om.ont import get_n, tokenize

from tqdm.auto import tqdm
from itertools import chain


def get_ld(ld, g, res, l=0):
    if type(ld) is tuple:
        res.append(' '.join(tokenize(get_n(ld[0], g))))
        for c in ld[1]:
            get_ld(c, g, res, l + 1)

    elif type(ld) is URIRef:
        res.append(str(g.value(ld, RDFS.label)))


def gen_embs(model, i, a, device):
    i = i.to(device)
    a = a.to(device)
    output = model(i, attention_mask=a, output_hidden_states=True)
    out = output['last_hidden_state'] * a.unsqueeze(-1)
    return out.sum(1) / a.sum(1, keepdim=True)


def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='LD similarity.')

    arg_parser.add_argument('source_ontology', help='Source ontology in order to retrieve labels.')
    arg_parser.add_argument('logical_definitions', help='Logic definitions in csv format.')
    arg_parser.add_argument('--output', dest='output', default='./output', help='Folder to save the results.')

    arg_parser.add_argument('--model', dest='model', nargs='?', default='dmis-lab/biobert-v1.1',
                            help='Model used to generate embeddings. Any model from huggingface transformers is expected to work.')
    arg_parser.add_argument('--aggregate', dest='aggregate', nargs='?', default='sep',
                            choices=['individual', 'mean', 'sep'],
                            help='The aggregation mode for the logical definitions. If individual, each logical definition entity is embedded separately and the final similarity is the average of the similarities between entity and each LD entity similarity. If mean, the embeddings of the LD entities are averaged. If sep, all entity labels are joined in a single string separated by [SEP] token.')
    arg_parser.add_argument('--return', dest='return_mode', nargs='?', default='similarity',
                            choices=['similarity', 'vectors'],
                            help='The return type. If similarity, the similarity between the entity and the LD is returned. If vectors, the entities and LDs embeddings are returned.')
    arg_parser.add_argument('--batch_size', dest='batch_size', nargs='?', type=int, default=32,
                            help='The batch size for the embeddings generation.')

    arg_parser.add_argument('--device', dest='device', nargs='?', default='cpu',
                            help='The device to use for the embeddings generation.')

    return arg_parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()

    o1 = Graph().parse(args.source_ontology)
    logdefs_raw = pd.read_csv(args.logical_definitions, on_bad_lines='skip', header=None, names=['e1', 'l1'])
    logdefs_raw = logdefs_raw.head(5)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.to(device)
    model.eval()

    x = []
    y = []
    for i, e1, l1 in logdefs_raw.itertuples(name=None):
        try:
            res = []
            get_ld(parser.parse(l1), o1, res)
            x.append(o1.value(URIRef(e1), RDFS.label, None, default=None))
            y.append('[SEP]'.join(res) if args.aggregate == 'sep' else res)
        except:
            logdefs_raw.drop(logdefs_raw.index[i])

    ex = tokenizer(x, return_tensors='pt', padding=True)
    xinput_ids = ex['input_ids']
    xattention_mask = ex['attention_mask']

    if args.aggregate == 'sep':
        ey = tokenizer(y, return_tensors='pt', padding=True)
        yinput_ids = ey['input_ids']
        yattention_mask = ey['attention_mask']
        myx = yinput_ids
    else:

        max_len = max([len(i) for i in y])

        my = [q + ['[PAD]'] * (max_len - len(q)) for q in y]
        myx = torch.Tensor([[1] * len(q) + [0] * (max_len - len(q)) for q in y])
        ny = list(chain(*my))

        ey = tokenizer(ny, return_tensors='pt', padding=True)
        yinput_ids = ey['input_ids']
        yattention_mask = ey['attention_mask']

        yinput_ids = yinput_ids.reshape((-1, max_len, yinput_ids.shape[1]))
        yattention_mask = yattention_mask.reshape((-1, max_len, yattention_mask.shape[1]))

    sims = []

    fembx = []
    femby = []

    with torch.no_grad():

        for x, ax, y, ay, ym in tqdm(
                DataLoader(TensorDataset(xinput_ids, xattention_mask, yinput_ids, yattention_mask, myx),
                           batch_size=args.batch_size)):

            xemb = gen_embs(model, x, ax, device)

            if args.return_mode == 'vectors':
                fembx.append(xemb.cpu())

            if args.aggregate == 'sep':

                emb = gen_embs(model, y, ay, device)
                if args.return_mode == 'vectors':
                    femby.append(emb.cpu())
                fs = torch.cosine_similarity(xemb, emb, dim=-1)

            else:

                fy = torch.flatten(y, end_dim=1)
                fay = torch.flatten(ay, end_dim=1)
                emb = gen_embs(model, fy, fay, device)
                nemb = emb.reshape(y.shape[0], y.shape[1], -1)

                if args.return_mode == 'vectors':
                    femby.append(nemb)

                if args.aggregate == 'individual':
                    csim = torch.cosine_similarity(xemb.unsqueeze(1), nemb, dim=-1)
                    fs = csim * ym
                    fs = fs.sum(1) / ym.sum(1)

                elif args.aggregate == 'mean':
                    nemb = (nemb * ym.unsqueeze(-1)).sum(1) / ym.sum(1, keepdim=True)
                    fs = torch.cosine_similarity(xemb, nemb, dim=-1)

            sims.append(fs.cpu())

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    if args.return_mode == 'similarity':
        logdefs_raw['similarity'] = torch.cat(sims).tolist()
        logdefs_raw.to_csv(os.path.join(args.output, 'similarity.csv'), index=False)

    elif args.return_mode == 'vectors':
        logdefs_raw.to_csv(os.path.join(args.output, 'logdefs.csv'), index=False)
        np.save(os.path.join(args.output, 'entities.npy'), torch.cat(fembx).numpy())
        np.save(os.path.join(args.output, 'logdefs.npy'), torch.cat(femby).numpy())
