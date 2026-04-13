"""
PU Learning Baselines (PULL, nnPU) for HeteroPULL paper.

동질 그래프(to_homogeneous) + GCN 인코더 기반 PU 학습 baseline.
기존 PULL/nnPU의 원래 설정(동질 그래프, GCN)을 따르되,
데이터 분할과 평가 방식은 HeteroPULL과 동일하게 맞춤.

사용법:
    python main_pu_baseline.py --method pull --seed 0 --gpu 0 \
        --result_json results/baseline_pull/seed_0.json

    python main_pu_baseline.py --method nnpu --seed 0 --gpu 0 \
        --result_json results/baseline_nnpu/seed_0.json
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
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score


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
    p.add_argument('--method', type=str, default='nnpu',
                   choices=['nnpu', 'pull'],
                   help='PU learning 방법 선택')
    # nnPU hyperparameters
    p.add_argument('--prior', type=float, default=0.004,
                   help='클래스 사전확률 pi (treats 비율 ≈ 0.4%%)')
    # PULL hyperparameters
    p.add_argument('--growth_rate', type=float, default=0.03)
    p.add_argument('--max_edge_ratio', type=float, default=1.0)
    p.add_argument('--confidence_threshold', type=float, default=0.85)
    p.add_argument('--result_json', type=str, default=None)
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ──────────────────────────────────────────────
# GCN 기반 동질 그래프 모델 (main_gcn_baseline.py와 동일 구조)
# ──────────────────────────────────────────────

class HomogeneousGCN(nn.Module):
    def __init__(self, data_hetero, hidden=128, out=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.dropout = dropout

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
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden, hidden))
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
        return self.out_lin(x)

    def decode(self, z_homo, edge_label_index, c_offset, d_offset):
        c = z_homo[edge_label_index[0] + c_offset]
        d = z_homo[edge_label_index[1] + d_offset]
        return (c * d).sum(dim=-1)

    def score_all(self, z_homo, c_offset, d_offset, n_c, n_d):
        z_c = z_homo[c_offset:c_offset + n_c]
        z_d = z_homo[d_offset:d_offset + n_d]
        return z_c @ z_d.t()


def homogenize_edges(data_hetero, type_offset, device):
    edge_list = []
    for (src, _, dst), ei in data_hetero.edge_index_dict.items():
        src_off = type_offset[src]
        dst_off = type_offset[dst]
        h = ei.clone()
        h[0] = h[0] + src_off
        h[1] = h[1] + dst_off
        edge_list.append(h)
    return torch.cat(edge_list, dim=1).to(device)


# ──────────────────────────────────────────────
# nnPU Loss
# ──────────────────────────────────────────────

def nnpu_loss(pos_logits, unl_logits, prior):
    """
    nnPU loss (Kiryo et al., 2017).
    R_pu = pi * R_p^+ + max(0, R_u^- - pi * R_p^-)
    """
    pos_pos_risk = F.binary_cross_entropy_with_logits(
        pos_logits, torch.ones_like(pos_logits), reduction='mean')
    pos_neg_risk = F.binary_cross_entropy_with_logits(
        pos_logits, torch.zeros_like(pos_logits), reduction='mean')
    unl_neg_risk = F.binary_cross_entropy_with_logits(
        unl_logits, torch.zeros_like(unl_logits), reduction='mean')

    neg_risk = unl_neg_risk - prior * pos_neg_risk
    neg_risk = torch.clamp(neg_risk, min=0.0)
    return prior * pos_pos_risk + neg_risk


# ──────────────────────────────────────────────
# PULL Graph Expansion (단일 loss, soft weight 없음)
# ──────────────────────────────────────────────

@torch.no_grad()
def pull_expand(z_homo, orig_edge, epoch, c_off, d_off, n_c, n_d,
                growth_rate, max_edge_ratio, confidence_threshold):
    """
    PULL의 Expected Graph 확장.
    HeteroPULL과 달리: soft weight 없음, 단일 BCE loss.
    """
    device = orig_edge.device
    z_c = z_homo[c_off:c_off + n_c]
    z_d = z_homo[d_off:d_off + n_d]
    scores = z_c @ z_d.t()
    probs = torch.sigmoid(scores)

    # 이미 알려진 positive 제외
    probs[orig_edge[0], orig_edge[1]] = 0.0

    n_orig = orig_edge.shape[1]
    target_n = int(n_orig * growth_rate * (epoch - 1))
    max_allowed = int(n_orig * max_edge_ratio)
    n_add = min(target_n, max_allowed)

    if n_add <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    flat_probs = probs.flatten()
    above_thr = (flat_probs > confidence_threshold).sum().item()
    n_add = min(n_add, int(above_thr))

    if n_add <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    topk_vals, topk_idx = torch.topk(flat_probs, n_add)
    rows = torch.div(topk_idx, n_d, rounding_mode='floor')
    cols = topk_idx % n_d
    return torch.stack([rows, cols], dim=0)


# ──────────────────────────────────────────────
# Filtered Ranking (동질 그래프용)
# ──────────────────────────────────────────────

@torch.no_grad()
def filtered_ranking(probs, pos_index, train_pos, val_pos, test_pos):
    """Compound→Disease 방향 filtered ranking."""
    all_known = torch.cat([train_pos, val_pos, test_pos], dim=1)
    ranks = []
    for i in range(pos_index.shape[1]):
        c, d = pos_index[0, i].item(), pos_index[1, i].item()
        row = probs[c].clone()
        mask = (all_known[0] == c)
        known_d = all_known[1][mask]
        row[known_d] = -float('inf')
        row[d] = probs[c, d]
        rank = (row > row[d]).sum().item() + 1
        ranks.append(rank)
    return np.array(ranks)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")
    print(f"PU 방법: {args.method.upper()}")

    data_path = osp.join('data', 'hetionet_data.pt')
    try:
        data = torch.load(data_path, weights_only=False)
    except TypeError:
        data = torch.load(data_path)

    transform = RandomLinkSplit(
        num_val=0.1, num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=1.0,
        edge_types=[EDGE_TYPE],
        rev_edge_types=[REV_EDGE_TYPE],
    )
    train_data, val_data, test_data = transform(data)

    def _split_edges(sd):
        return {
            'index': sd[EDGE_TYPE].edge_label_index,
            'label': sd[EDGE_TYPE].edge_label,
        }

    val_edges = _split_edges(val_data)
    test_edges = _split_edges(test_data)

    data = data.to(device)
    train_data = train_data.to(device)
    for s in (val_edges, test_edges):
        s['index'] = s['index'].to(device)
        s['label'] = s['label'].to(device)

    model = HomogeneousGCN(
        data, hidden=args.hidden_dim, out=args.out_dim,
        num_layers=args.layers, dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    c_off = model.type_offset['Compound']
    d_off = model.type_offset['Disease']
    n_c = data['Compound'].num_nodes
    n_d = data['Disease'].num_nodes

    train_homo_edge = homogenize_edges(train_data, model.type_offset, device)
    train_pos = train_data[EDGE_TYPE].edge_index  # [2, N_train] (hetero index)

    pos_mask_val = val_edges['label'] > 0.5
    pos_mask_test = test_edges['label'] > 0.5
    val_pos_only = val_edges['index'][:, pos_mask_val]
    test_pos_only = test_edges['index'][:, pos_mask_test]

    @torch.no_grad()
    def eval_auc(z_homo, edges):
        logits = model.decode(z_homo, edges['index'], c_off, d_off)
        p = torch.sigmoid(logits).cpu().numpy()
        l = edges['label'].cpu().numpy()
        return roc_auc_score(l, p), average_precision_score(l, p)

    best_val_auc = -1.0
    best_state = None
    best_epoch = 0
    patience_counter = 0
    z_prev = None

    print(f"\n[{args.method.upper()} Baseline 학습 시작]")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()

        # PULL: graph expansion (epoch >= 2)
        if args.method == 'pull' and epoch > 1 and z_prev is not None:
            exp_edge = pull_expand(
                z_prev, train_pos, epoch, c_off, d_off, n_c, n_d,
                args.growth_rate, args.max_edge_ratio,
                args.confidence_threshold)
            current_pos = torch.cat([train_pos, exp_edge], dim=1)
        else:
            current_pos = train_pos
            exp_edge = torch.empty((2, 0), dtype=torch.long, device=device)

        optimizer.zero_grad()
        x = model.build_init_x(device)
        z = model.encode(x, train_homo_edge)
        z_prev = z.detach()

        pos_logits = model.decode(z, current_pos, c_off, d_off)

        if args.method == 'nnpu':
            # Unlabeled: 랜덤 (Compound, Disease) 쌍
            unl_src = torch.randint(0, n_c, (current_pos.size(1) * 2,), device=device)
            unl_dst = torch.randint(0, n_d, (current_pos.size(1) * 2,), device=device)
            unl_edge = torch.stack([unl_src, unl_dst], dim=0)
            unl_logits = model.decode(z, unl_edge, c_off, d_off)
            loss = nnpu_loss(pos_logits, unl_logits, args.prior)

        elif args.method == 'pull':
            # PULL: 단일 BCE, soft weight 없음
            neg_src = torch.randint(0, n_c, (current_pos.size(1),), device=device)
            neg_dst = torch.randint(0, n_d, (current_pos.size(1),), device=device)
            neg_edge = torch.stack([neg_src, neg_dst], dim=0)
            neg_logits = model.decode(z, neg_edge, c_off, d_off)

            all_logits = torch.cat([pos_logits, neg_logits])
            all_labels = torch.cat([
                torch.ones(pos_logits.size(0), device=device),
                torch.zeros(neg_logits.size(0), device=device),
            ])
            loss = F.binary_cross_entropy_with_logits(all_logits, all_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Eval
        model.eval()
        with torch.no_grad():
            x_e = model.build_init_x(device)
            z_e = model.encode(x_e, train_homo_edge)
        val_auc, val_auprc = eval_auc(z_e, val_edges)
        test_auc, test_auprc = eval_auc(z_e, test_edges)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stop at epoch {epoch}")
                break

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss {loss.item():.4f} | "
                  f"Exp {exp_edge.size(1)} | "
                  f"Val AUC {val_auc:.4f} AUPRC {val_auprc:.4f} | "
                  f"Test AUC {test_auc:.4f} AUPRC {test_auprc:.4f}")

    elapsed = time.time() - start_time
    print(f"\n총 시간: {elapsed:.1f}s, Best epoch {best_epoch}")

    # ── Final eval ──
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        x_f = model.build_init_x(device)
        z_f = model.encode(x_f, train_homo_edge)

    val_auc, val_auprc = eval_auc(z_f, val_edges)
    test_auc, test_auprc = eval_auc(z_f, test_edges)

    scores = model.score_all(z_f, c_off, d_off, n_c, n_d)
    probs = torch.sigmoid(scores)

    test_ranks = filtered_ranking(probs, test_pos_only, train_pos, val_pos_only, test_pos_only)
    val_ranks = filtered_ranking(probs, val_pos_only, train_pos, val_pos_only, test_pos_only)

    def _metrics(ranks):
        r = ranks.astype(float)
        return {
            'MRR': float(np.mean(1.0 / r)),
            'Hits@1': float(np.mean(r <= 1)),
            'Hits@3': float(np.mean(r <= 3)),
            'Hits@10': float(np.mean(r <= 10)),
        }

    val_rank = _metrics(val_ranks)
    test_rank = _metrics(test_ranks)

    print(f"\n[Final — {args.method.upper()} Baseline]")
    print(f"Val  AUC={val_auc:.4f} AUPRC={val_auprc:.4f} "
          f"MRR={val_rank['MRR']:.4f} H@3={val_rank['Hits@3']:.3f} "
          f"H@10={val_rank['Hits@10']:.3f}")
    print(f"Test AUC={test_auc:.4f} AUPRC={test_auprc:.4f} "
          f"MRR={test_rank['MRR']:.4f} H@3={test_rank['Hits@3']:.3f} "
          f"H@10={test_rank['Hits@10']:.3f}")

    if args.result_json:
        out_dir = osp.dirname(args.result_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = {
            'model': f'{args.method.upper()} Baseline',
            'seed': args.seed,
            'best_epoch': best_epoch,
            'final': {
                'val_auc': float(val_auc), 'val_auprc': float(val_auprc),
                'val_mrr': float(val_rank['MRR']),
                'val_h1': float(val_rank['Hits@1']),
                'val_h3': float(val_rank['Hits@3']),
                'val_h10': float(val_rank['Hits@10']),
                'test_auc': float(test_auc), 'test_auprc': float(test_auprc),
                'test_mrr': float(test_rank['MRR']),
                'test_h1': float(test_rank['Hits@1']),
                'test_h3': float(test_rank['Hits@3']),
                'test_h10': float(test_rank['Hits@10']),
            },
        }
        with open(args.result_json, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"[result_json 저장] {args.result_json}")


if __name__ == '__main__':
    main()
