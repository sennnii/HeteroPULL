"""
Degree-based interpretability analysis for HeteroPULL.

각 test positive (compound, disease) 엣지에 대해 filtered rank를 계산하고,
compound/disease의 훈련 그래프 내 degree 기준으로 low/mid/high 버킷으로 나눠
버킷별 MRR, H@10을 비교한다.

HeteroPULL과 HGT baseline 두 모델을 학습/평가하여 비교.

사용법:
    python scripts/analyze_degree.py --seed 0 --gpu 0 \
        --out_json results/interpret/degree.json
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
from torch_geometric.transforms import RandomLinkSplit

from src.model_hetionet import HeteroPULLModel
from src.train_hetionet import train, test


EDGE_TYPE = ('Compound', 'treats', 'Disease')
REV_EDGE_TYPE = ('Disease', 'rev_treats', 'Compound')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--out_dim', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--lr', type=float, default=0.003)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--inner_steps', type=int, default=50)
    p.add_argument('--growth_rate', type=float, default=0.03)
    p.add_argument('--max_edge_ratio', type=float, default=1.0)
    p.add_argument('--confidence_threshold', type=float, default=0.85)
    p.add_argument('--lambda_c', type=float, default=1.0)
    p.add_argument('--out_json', type=str, default='results/interpret/degree.json')
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_heteropull(args, data, train_data, val_edges, eval_edge_index_dict, device):
    """HeteroPULL 학습 (main_hetionet.py 핵심 로직만 발췌)."""
    model = HeteroPULLModel(
        data=data,
        hidden_channels=args.hidden_dim,
        out_channels=args.out_dim,
        num_heads=args.heads,
        num_layers=args.layers,
        use_compound_features=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    best_val_auc = -1.0
    best_state = None
    patience = 0
    z_dict_prev = None

    for epoch in range(1, args.epochs + 1):
        loss, z_dict_prev, _ = train(
            model, optimizer, data, train_data, epoch,
            z_dict_prev=z_dict_prev,
            inner_steps=args.inner_steps,
            growth_rate=args.growth_rate,
            max_edge_ratio=args.max_edge_ratio,
            confidence_threshold=args.confidence_threshold,
            lambda_c=args.lambda_c,
        )
        _, val_auc, _ = test(data, model, val_edges, eval_edge_index_dict)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break
        if epoch % 5 == 0:
            print(f'  Epoch {epoch:02d} loss={loss:.4f} val_auc={val_auc:.4f}')

    model.load_state_dict(best_state)
    return model


def train_hgt_baseline(args, data, train_data, val_edges, eval_edge_index_dict, device):
    """HGT baseline: HeteroPULL과 동일 구조지만 PU 확장/dual-loss 없음."""
    args_copy = argparse.Namespace(**vars(args))
    args_copy.growth_rate = 0.0
    args_copy.lambda_c = 0.0
    return train_heteropull(args_copy, data, train_data, val_edges, eval_edge_index_dict, device)


@torch.no_grad()
def compute_ranks_and_degrees(model, data, eval_edge_index_dict,
                               test_pos, train_pos, val_pos, device):
    """각 test positive edge에 대해 filtered rank + compound/disease degree 계산."""
    model.eval()
    z_dict = model.encode(data, eval_edge_index_dict, None)
    scores = z_dict['Compound'] @ z_dict['Disease'].t()
    probs = torch.sigmoid(scores)
    n_c, n_d = scores.shape

    # Filter mask: 모든 known positive
    filter_mask = torch.zeros((n_c, n_d), dtype=torch.bool, device=device)
    for pe in (train_pos, val_pos, test_pos):
        if pe is not None and pe.numel() > 0:
            filter_mask[pe[0], pe[1]] = True

    # Compound degree (training treats)
    # 여기서는 전체 train 엣지가 아닌 treats 엣지에 대한 degree
    compound_degree = torch.zeros(n_c, dtype=torch.long, device=device)
    disease_degree = torch.zeros(n_d, dtype=torch.long, device=device)
    compound_degree.scatter_add_(0, train_pos[0], torch.ones_like(train_pos[0]))
    disease_degree.scatter_add_(0, train_pos[1], torch.ones_like(train_pos[1]))

    records = []
    for i in range(test_pos.size(1)):
        c = test_pos[0, i].item()
        d = test_pos[1, i].item()

        # Disease-ranking: compound 고정
        row = probs[c].clone()
        target = row[d].item()
        row[filter_mask[c]] = float('-inf')
        row[d] = target
        rank_d = (row > target).sum().item() + 1

        # Compound-ranking: disease 고정
        col = probs[:, d].clone()
        target2 = col[c].item()
        col[filter_mask[:, d]] = float('-inf')
        col[c] = target2
        rank_c = (col > target2).sum().item() + 1

        records.append({
            'compound': c,
            'disease': d,
            'compound_degree': int(compound_degree[c]),
            'disease_degree': int(disease_degree[d]),
            'rank_tail': rank_d,  # disease를 ranking
            'rank_head': rank_c,  # compound를 ranking
        })
    return records


def bucket_metrics(records, key):
    """key 기준으로 low/mid/high 버킷 나누고 MRR, H@10 계산."""
    degrees = sorted([r[key] for r in records])
    n = len(degrees)
    if n == 0:
        return {}
    # Tercile 분할
    t1 = degrees[n // 3]
    t2 = degrees[(2 * n) // 3]

    def bucket_name(d):
        if d <= t1:
            return 'low'
        elif d <= t2:
            return 'mid'
        else:
            return 'high'

    buckets = {'low': [], 'mid': [], 'high': []}
    for r in records:
        b = bucket_name(r[key])
        # tail과 head rank 모두 사용
        buckets[b].append(r['rank_tail'])
        buckets[b].append(r['rank_head'])

    result = {}
    for b, ranks in buckets.items():
        if not ranks:
            result[b] = {'n': 0}
            continue
        ranks_arr = np.array(ranks, dtype=float)
        result[b] = {
            'n': len(ranks),
            'MRR': float(np.mean(1.0 / ranks_arr)),
            'H@1': float(np.mean(ranks_arr <= 1)),
            'H@3': float(np.mean(ranks_arr <= 3)),
            'H@10': float(np.mean(ranks_arr <= 10)),
            'mean_degree_range': [int(min(r[key] for r in records if bucket_name(r[key]) == b)),
                                   int(max(r[key] for r in records if bucket_name(r[key]) == b))],
        }
    return result


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Data loading
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

    def _split(sd):
        return {'index': sd[EDGE_TYPE].edge_label_index,
                'label': sd[EDGE_TYPE].edge_label}

    val_edges = _split(val_data)
    test_edges = _split(test_data)

    pos_mask_v = val_edges['label'] > 0.5
    pos_mask_t = test_edges['label'] > 0.5

    data = data.to(device)
    train_data = train_data.to(device)
    for s in (val_edges, test_edges):
        s['index'] = s['index'].to(device)
        s['label'] = s['label'].to(device)

    train_pos = train_data[EDGE_TYPE].edge_index
    val_pos = val_edges['index'][:, pos_mask_v]
    test_pos = test_edges['index'][:, pos_mask_t]
    eval_edge_index_dict = {k: v for k, v in train_data.edge_index_dict.items()}

    # Train HeteroPULL
    print('\n[HeteroPULL 학습]')
    t0 = time.time()
    model_ours = train_heteropull(args, data, train_data, val_edges, eval_edge_index_dict, device)
    print(f'  학습 시간: {time.time()-t0:.1f}s')

    print('\n[HGT baseline 학습]')
    t0 = time.time()
    model_hgt = train_hgt_baseline(args, data, train_data, val_edges, eval_edge_index_dict, device)
    print(f'  학습 시간: {time.time()-t0:.1f}s')

    # Analysis
    print('\n[Rank + degree 계산]')
    records_ours = compute_ranks_and_degrees(
        model_ours, data, eval_edge_index_dict, test_pos, train_pos, val_pos, device)
    records_hgt = compute_ranks_and_degrees(
        model_hgt, data, eval_edge_index_dict, test_pos, train_pos, val_pos, device)

    # Bucket by compound degree and disease degree
    print('\n[버킷별 성능 (compound degree)]')
    ours_by_c = bucket_metrics(records_ours, 'compound_degree')
    hgt_by_c = bucket_metrics(records_hgt, 'compound_degree')
    for b in ['low', 'mid', 'high']:
        if b in ours_by_c and 'MRR' in ours_by_c[b]:
            print(f'  {b:>4s} (deg={ours_by_c[b]["mean_degree_range"]}, n={ours_by_c[b]["n"]}): '
                  f'HeteroPULL MRR={ours_by_c[b]["MRR"]:.4f} H@10={ours_by_c[b]["H@10"]:.3f} | '
                  f'HGT MRR={hgt_by_c[b]["MRR"]:.4f} H@10={hgt_by_c[b]["H@10"]:.3f}')

    print('\n[버킷별 성능 (disease degree)]')
    ours_by_d = bucket_metrics(records_ours, 'disease_degree')
    hgt_by_d = bucket_metrics(records_hgt, 'disease_degree')
    for b in ['low', 'mid', 'high']:
        if b in ours_by_d and 'MRR' in ours_by_d[b]:
            print(f'  {b:>4s} (deg={ours_by_d[b]["mean_degree_range"]}, n={ours_by_d[b]["n"]}): '
                  f'HeteroPULL MRR={ours_by_d[b]["MRR"]:.4f} H@10={ours_by_d[b]["H@10"]:.3f} | '
                  f'HGT MRR={hgt_by_d[b]["MRR"]:.4f} H@10={hgt_by_d[b]["H@10"]:.3f}')

    # Save
    out_dir = osp.dirname(args.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = {
        'seed': args.seed,
        'n_test_edges': int(test_pos.size(1)),
        'by_compound_degree': {'HeteroPULL': ours_by_c, 'HGT': hgt_by_c},
        'by_disease_degree': {'HeteroPULL': ours_by_d, 'HGT': hgt_by_d},
        'records_ours': records_ours,
        'records_hgt': records_hgt,
    }
    with open(args.out_json, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'\n[저장] {args.out_json}')


if __name__ == '__main__':
    main()
