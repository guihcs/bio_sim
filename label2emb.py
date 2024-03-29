import os

import argparse

import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm


def gen_embs(model, i, a, device):
    i = i.to(device)
    a = a.to(device)
    output = model(i, attention_mask=a, output_hidden_states=True)
    out = output['last_hidden_state'] * a.unsqueeze(-1)
    return out.sum(1) / a.sum(1, keepdim=True)


def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='Get embeddings from labels.')

    arg_parser.add_argument('labels', help='Source labels.')
    arg_parser.add_argument('--output', dest='output', default='./output', help='Folder to save the results.')

    arg_parser.add_argument('--model', dest='model', nargs='?', default='dmis-lab/biobert-v1.1',
                            help='Model used to generate embeddings. Any model from huggingface transformers is expected to work.')

    arg_parser.add_argument('--batch_size', dest='batch_size', nargs='?', type=int, default=32,
                            help='The batch size for the embeddings generation.')

    arg_parser.add_argument('--device', dest='device', nargs='?', default='cpu',
                            help='The device to use for the embeddings generation.')

    arg_parser.add_argument('--state', dest='state', nargs='?', help='State of already trained model.')

    return arg_parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)

    if args.state:
        model.load_state_dict(torch.load(args.state, map_location=device))

    model.to(device)
    model.eval()

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    with open(args.labels, 'rb') as f:
        num_lines = sum(1 for _ in f)

    with open(args.labels) as fr:
        with open(os.path.join(args.output, 'embeddings.txt'), 'w') as f:

            batch = []
            for line in tqdm(fr, total=num_lines):
                batch.append(line.strip())
                if len(batch) == args.batch_size:

                    ex = tokenizer(batch, return_tensors='pt', padding=True)
                    xinput_ids = ex['input_ids']
                    xattention_mask = ex['attention_mask']

                    with torch.no_grad():

                        for q, i, a in DataLoader(TensorDataset(torch.arange(0, len(batch)), xinput_ids, xattention_mask), batch_size=args.batch_size):
                            embedding = gen_embs(model, i, a, device).cpu()
                            elines = []
                            for qw, e in zip(q, embedding.tolist()):
                                elines.append(batch[qw.item()] + ',' + ' '.join([str(x) for x in e]))
                        f.write('\n'.join(elines) + '\n')
                    batch = []
                    f.flush()