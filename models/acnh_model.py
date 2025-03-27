import torch
import torch.nn as nn

class ACNHModel(nn.Module):
    def __init__(self, out_features, emb_sizes, layers, drop_p):
        super().__init__()
        self.embed = nn.ModuleList([nn.Embedding(ni, nf) for (ni, nf) in emb_sizes])
        self.embed_drop = nn.Dropout(p=drop_p)

        n_in = sum([nf for _, nf in emb_sizes])

        layerlist = []
        for units in layers:
            layerlist.append(nn.Linear(n_in, units))
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(units))
            layerlist.append(nn.Dropout(p=drop_p))
            n_in = units

        layerlist.append(nn.Linear(layers[-1], out_features))

        self.layers = nn.Sequential(*layerlist)

    def forward(self, x_feats):
        embeddings = []

        for i, e in enumerate(self.embed):
            embeddings.append(e(x_feats[:, i]))

        x = torch.concat(embeddings, dim=1)
        x = self.embed_drop(x)
        x = self.layers(x)

        return x
        