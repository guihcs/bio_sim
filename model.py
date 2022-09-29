import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
        self.biobert = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
        for param in self.biobert.base_model.parameters():
            param.requires_grad = False

        self.biobert.eval()

        self.dff = nn.Sequential(
            nn.Linear(768 * 2, 768 * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768 * 4, 768),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def emb(self, x, a):
        out = self.biobert(input_ids=x, attention_mask=a)['last_hidden_state']
        cf = a.sum(dim=1, keepdims=True)
        cf[cf == 0] = 1
        res = out.sum(1) / cf
        return res

    def forward(self, x, xa, y, ya):
        e1 = self.emb(x, xa)
        bs = y.shape[0]
        ss = y.shape[1]

        e2 = self.emb(torch.flatten(y, end_dim=1).long(), torch.flatten(ya, end_dim=1))

        jc = torch.cat([e1.unsqueeze(1).repeat(1, ss, 1), e2.reshape(bs, ss, -1)], dim=2)

        return self.dff(jc)


    def sims(self, anc, ex, bs=10, device=torch.device('cpu')):
        at = self.tokenizer([anc] + ex, return_tensors='pt', padding=True)


        ids = at['input_ids'].to(device)
        ats = at['attention_mask'].to(device)

        act = ids[0]
        aca = ats[0]

        ext = ids[1:]
        exa = ats[1:]


        os = []

        with torch.no_grad():

            for e, a in DataLoader(list(zip(ext, exa)), batch_size=bs):
                out = self(act.unsqueeze(0), aca.unsqueeze(0), e.unsqueeze(0), a.unsqueeze(0))
                os.append(out.squeeze(0).t().squeeze(0))

        return torch.cat(os, dim=0).cpu()
