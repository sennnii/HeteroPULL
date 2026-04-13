"""
GCN Baseline for HeteroPULL paper.
이질 그래프를 to_homogeneous()로 동종화한 뒤 GCN으로 link prediction.
사용법:
    python main_gcn_baseline.py --seed 0 --gpu 0 \
        --result_json results/gcn_baseline/seed0.json
"""
import os
import os.path as osp
import argparse
import copy
import json
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score

from src.train_hetionet import evaluate_ranking


EDGE_TYPE = ('Compound', 'treats', 'Disease')
REV_EDGE_TYPE = ('Disease', 'rev_treats', 'Compound')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--out_dim', type=int, default=64)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--model', type=str, default='gcn',
                   choices=['gcn', 'gat', 'sage'],
                   help='baseline 모델 선택')
    p.add_argument('--result_json', type=str, default=None)
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class HomogeneousBaseline(nn.Module):
    """동종 그래프에서 동작하는 link prediction baseline (GCN/GAT/SAGE)."""
    def __init__(self, data_hetero, hidden=128, out=64, num_layers=2,
                 dropout=0.3, model_type='gcn'):
        super().__init__()
        self.dropout = dropout
        self.model_type = model_type
        # 각 node type 을 learnable embedding 으로 동일하게 처리 (공정 비교를 위해
        # compound feature 도 사용하지 않음 — 즉 순수 구조 정보만 사용)
        self.node_embs = nn.ModuleDict()
        self.type_offset = {}
        offset = 0
        for nt in data_hetero.node_types:
            n = data_hetero[nt].num_nodes
            self.node_embs[nt] = nn.Embedding(n, hidden)
            self.type_offset[nt] = offset
            offset += n
        self.total_nodes = offset

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if model_type == 'gcn':
                self.convs.append(GCNConv(hidden, hidden))
            elif model_type == 'gat':
                self.convs.append(GATConv(hidden, hidden, heads=4, concat=False))
            elif model_type == 'sage':
                self.convs.append(SAGEConv(hidden, hidden))
            else:
                raise ValueError(f'Unknown model_type: {model_type}')
        self.out_lin = nn.Linear(hidden, out)

        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.node_embs.values():
            nn.init.xavier_uniform_(emb.weight)

    def build_init_x(self, device):
        xs = []
        for nt in self.node_embs:
            xs.append(self.node_embs[nt].weight)
        return torch.cat(xs, dim=0).to(device)

    def encode(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out_lin(x)
        return x

    def decode(self, z_homo, edge_label_index, c_offset, d_offset):
        c = z_homo[edge_label_index[0] + c_offset]
        d = z_homo[edge_label_index[1] + d_offset]
        return (c * d).sum(dim=-1)


def homogenize_edges(data_hetero, c_offset_map, device):
    """이질 엣지들을 동종 edge_index 로 merge."""
    edge_list = []
    for (src, _, dst), ei in data_hetero.edge_index_dict.items():
        src_off = c_offset_map[src]
        dst_off = c_offset_map[dst]
        homogenized = ei.clone()
        homogenized[0] = homogenized[0] + src_off
        homogenized[1] = homogenized[1] + dst_off
        edge_list.append(homogenized)
    return torch.cat(edge_list, dim=1).to(device)


@torch.no_grad()
def eval_auc_auprc(model, z_homo, edges, c_off, d_off):
    logits = model.decode(z_homo, edges['index'], c_off, d_off)
    probs = torch.sigmoid(logits).cpu().numpy()
    labels = edges['label'].cpu().numpy()
    auc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)
    return auc, auprc


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")

    data_path = osp.join('data', 'hetionet_data.pt')
    print("Hetionet 데이터 로드 중...")
    try:
        data = torch.load(data_path, weights_only=False)
    except TypeError:
        data = torch.load(data_path)

    # Link split
    transform = RandomLinkSplit(
        num_val=0.1, num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=1.0,
        edge_types=[EDGE_TYPE],
        rev_edge_types=[REV_EDGE_TYPE],
    )
    train_data, val_data, test_data = transform(data)

    def _split_edges(split_data):
        return {
            'index': split_data[EDGE_TYPE].edge_label_index,
            'label': split_data[EDGE_TYPE].edge_label,
        }

    val_edges = _split_edges(val_data)
    test_edges = _split_edges(test_data)

    data = data.to(device)
    train_data = train_data.to(device)
    for split in (val_edges, test_edges):
        split['index'] = split['index'].to(device)
        split['label'] = split['label'].to(device)

    model = HomogeneousBaseline(
        data, hidden=args.hidden_dim, out=args.out_dim,
        num_layers=args.layers, dropout=args.dropout,
        model_type=args.model,
    ).to(device)
    print(f"Baseline model: {args.model.upper()}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    c_off = model.type_offset['Compound']
    d_off = model.type_offset['Disease']

    # Homogeneous edge_index for message passing (train 엣지만 사용)
    train_edge_index_homo = homogenize_edges(
        train_data, model.type_offset, device,
    )

    train_pos_index = train_data[EDGE_TYPE].edge_index  # [2, N_train]

    # Eval용 — train 엣지만 encoder 에 노출
    best_val_mrr = -1.0
    best_state = None
    best_epoch = 0
    patience = 0

    pos_mask_val = val_edges['label'] > 0.5
    pos_mask_test = test_edges['label'] > 0.5
    val_pos_only = val_edges['index'][:, pos_mask_val]
    test_pos_only = test_edges['index'][:, pos_mask_test]

    print("[GCN Baseline 학습 시작]")
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        x = model.build_init_x(device)
        z = model.encode(x, train_edge_index_homo)

        # Positive edges
        pos_logits = model.decode(z, train_pos_index, c_off, d_off)

        # Negative sampling
        num_c = data['Compound'].num_nodes
        num_d = data['Disease'].num_nodes
        neg_src = torch.randint(0, num_c, (train_pos_index.shape[1],), device=device)
        neg_dst = torch.randint(0, num_d, (train_pos_index.shape[1],), device=device)
        neg_edge = torch.stack([neg_src, neg_dst], dim=0)
        neg_logits = model.decode(z, neg_edge, c_off, d_off)

        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat([
            torch.ones_like(pos_logits),
            torch.zeros_like(neg_logits),
        ], dim=0)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Eval
        model.eval()
        with torch.no_grad():
            x_eval = model.build_init_x(device)
            z_eval = model.encode(x_eval, train_edge_index_homo)

            val_auc, val_auprc = eval_auc_auprc(model, z_eval, val_edges, c_off, d_off)
            test_auc, test_auprc = eval_auc_auprc(model, z_eval, test_edges, c_off, d_off)

        # Simple ranking eval: rank each val positive against all diseases
        # (간단화 — HeteroPULL 과 동일한 filtered ranking 사용은 구조가 달라 복잡,
        #  여기서는 AUC/AUPRC 위주로 비교)
        val_mrr_proxy = val_auc  # 간이 지표로 대체

        if val_auc > best_val_mrr:
            best_val_mrr = val_auc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_val_auc = val_auc
            best_val_auprc = val_auprc
            best_test_auc = test_auc
            best_test_auprc = test_auprc
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stop at epoch {epoch}")
                break

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss {loss.item():.4f} | "
                  f"Val AUC {val_auc:.4f} AUPRC {val_auprc:.4f} | "
                  f"Test AUC {test_auc:.4f} AUPRC {test_auprc:.4f}")

    print(f"\n총 시간: {time.time()-start_time:.1f}s, Best epoch {best_epoch}")
    print(f"Best Val  AUC {best_val_auc:.4f} AUPRC {best_val_auprc:.4f}")
    print(f"Best Test AUC {best_test_auc:.4f} AUPRC {best_test_auprc:.4f}")

    # Ranking metrics on best model
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        x_eval = model.build_init_x(device)
        z_eval = model.encode(x_eval, train_edge_index_homo)

    # Filtered ranking 계산 (HeteroPULL 과 동일한 방식)
    # 간이 래퍼: score_all
    class _Wrap:
        def __init__(self, z_c, z_d):
            self.z_c = z_c
            self.z_d = z_d
        def score_all(self, z_dict, src_type='Compound', dst_type='Disease'):
            return self.z_c @ self.z_d.t()

    z_c = z_eval[c_off:c_off + data['Compound'].num_nodes]
    z_d = z_eval[d_off:d_off + data['Disease'].num_nodes]

    # Compute ranking manually
    scores = z_c @ z_d.t()  # [C, D]
    probs = torch.sigmoid(scores)
    train_pos = train_pos_index

    # Disease-ranking for each test positive compound
    def _filtered_ranks(pos_index, train_pos, val_pos, test_pos, mode='test'):
        ranks = []
        all_known = torch.cat([train_pos, val_pos, test_pos], dim=1)
        for i in range(pos_index.shape[1]):
            c, d = pos_index[0, i].item(), pos_index[1, i].item()
            row = probs[c].clone()
            # Mask known positives except current
            mask = (all_known[0] == c)
            known_d = all_known[1][mask]
            row[known_d] = -float('inf')
            row[d] = probs[c, d]  # restore current
            rank = (row > row[d]).sum().item() + 1
            ranks.append(rank)
        return np.array(ranks)

    test_ranks = _filtered_ranks(test_pos_only, train_pos, val_pos_only, test_pos_only)
    test_mrr = float(np.mean(1.0 / test_ranks))
    test_h1 = float(np.mean(test_ranks <= 1))
    test_h3 = float(np.mean(test_ranks <= 3))
    test_h10 = float(np.mean(test_ranks <= 10))

    print(f"\n[Final Filtered Ranking]")
    print(f"Test MRR={test_mrr:.4f} H@1={test_h1:.3f} "
          f"H@3={test_h3:.3f} H@10={test_h10:.3f}")

    if args.result_json is not None:
        out_dir = osp.dirname(args.result_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = {
            'model': f'{args.model.upper()} Baseline',
            'seed': args.seed,
            'best_epoch': best_epoch,
            'final': {
                'val_auc': best_val_auc,
                'val_auprc': best_val_auprc,
                'test_auc': best_test_auc,
                'test_auprc': best_test_auprc,
                'test_mrr': test_mrr,
                'test_h1': test_h1,
                'test_h3': test_h3,
                'test_h10': test_h10,
            },
        }
        with open(args.result_json, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"[result_json 저장] {args.result_json}")


if __name__ == '__main__':
    main()
