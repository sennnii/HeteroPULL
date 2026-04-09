import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear


class HeteroPULLModel(nn.Module):
    def __init__(self, data, hidden_channels=128, out_channels=64,
                 num_heads=4, num_layers=2, dropout=0.3,
                 use_compound_features=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.use_compound_features = use_compound_features

        self.input_lins = nn.ModuleDict()
        self.node_embeds = nn.ModuleDict()
        for node_type in data.node_types:
            has_feat = (hasattr(data[node_type], 'x')
                        and data[node_type].x is not None)
            # Ablation: Compound feature(Morgan FP) 를 무시하고 learnable
            # embedding 만 쓰도록 강제.
            if node_type == 'Compound' and not use_compound_features:
                has_feat = False
            if has_feat:
                in_dim = data[node_type].x.shape[1]
                self.input_lins[node_type] = nn.Sequential(
                    Linear(in_dim, hidden_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            else:
                num_nodes = data[node_type].num_nodes
                self.node_embeds[node_type] = nn.Embedding(num_nodes, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HGTConv(hidden_channels, hidden_channels,
                        data.metadata(), num_heads)
            )

        self.lin_dict = nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(hidden_channels, out_channels)

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.node_embeds.values():
            nn.init.xavier_uniform_(emb.weight)

    def _initial_x(self, data):
        x_dict = {}
        for node_type in data.node_types:
            if node_type in self.input_lins:
                x_dict[node_type] = self.input_lins[node_type](data[node_type].x)
            else:
                x_dict[node_type] = self.node_embeds[node_type].weight
        return x_dict

    def encode(self, data, edge_index_dict, edge_weight_dict=None):
        x_dict = self._initial_x(data)
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {nt: self.dropout(F.elu(x)) for nt, x in x_dict.items()}

        # Final projection. L2 normalize 는 logit 을 cosine 유사도 [-1,1] 로
        # 압축해 BCE 의 초반 학습 속도를 떨어뜨리므로 사용하지 않는다.
        z_dict = {nt: lin(x_dict[nt]) for nt, lin in self.lin_dict.items()}
        return z_dict

    def decode(self, z_dict, edge_label_index,
               src_type='Compound', dst_type='Disease'):
        src = z_dict[src_type][edge_label_index[0]]
        dst = z_dict[dst_type][edge_label_index[1]]
        return (src * dst).sum(dim=-1)

    @torch.no_grad()
    def score_all(self, z_dict, src_type='Compound', dst_type='Disease'):
        return z_dict[src_type] @ z_dict[dst_type].t()
