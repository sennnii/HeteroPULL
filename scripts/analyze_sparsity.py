"""
Sparsity robustness experiment.

treats 엣지의 일정 비율(예: 25%, 50%, 75%, 100%)만 학습에 사용하여
HeteroPULL과 HGT baseline의 성능을 비교한다.
데이터 희소도가 높아질수록 HeteroPULL의 상대적 우위가 커지는지 확인.

사용법:
    python -m scripts.analyze_sparsity --seeds 0 1 2 --gpu 0 \
        --fractions 0.25 0.5 0.75 1.0 \
        --out_json results/interpret/sparsity.json
"""
import os
import os.path as osp
import argparse
import copy
import json
import random
import time
import math
import numpy as np
import torch
from torch_geometric.transforms import RandomLinkSplit

from src.model_hetionet import HeteroPULLModel
from src.train_hetionet import train, test, evaluate_ranking


EDGE_TYPE = ('Compound', 'treats', 'Disease')
REV_EDGE_TYPE = ('Disease', 'rev_treats', 'Compound')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--fractions', type=float, nargs='+', default=[0.25, 0.5, 0.75, 1.0])
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
    p.add_argument('--out_json', type=str,
                   default='results/interpret/sparsity.json')
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def subsample_treats(train_data, fraction, seed):
    """train_data의 treats 엣지를 fraction만큼 무작위 샘플링."""
    full_edge = train_data[EDGE_TYPE].edge_index
    n_full = full_edge.size(1)
    n_keep = max(1, int(n_full * fraction))
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n_full, generator=g)[:n_keep]
    kept = full_edge[:, perm]

    # train_data를 얕은 복제해서 treats만 교체
    new_train = copy.copy(train_data)
    new_train[EDGE_TYPE] = copy.copy(train_data[EDGE_TYPE])
    new_train[EDGE_TYPE].edge_index = kept
    # rev_treats 도 동기화
    new_train[REV_EDGE_TYPE] = copy.copy(train_data[REV_EDGE_TYPE])
    new_train[REV_EDGE_TYPE].edge_index = kept.flip(0)
    return new_train, n_keep


def train_model(args, data, train_data, val_edges, eval_edge_index_dict,
                device, use_pu=True):
    """HeteroPULL 또는 HGT baseline 학습."""
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

    # HGT baseline: growth_rate=0, lambda_c=0
    gr = args.growth_rate if use_pu else 0.0
    lc = args.lambda_c if use_pu else 0.0

    for epoch in range(1, args.epochs + 1):
        _, z_dict_prev, _ = train(
            model, optimizer, data, train_data, epoch,
            z_dict_prev=z_dict_prev,
            inner_steps=args.inner_steps,
            growth_rate=gr,
            max_edge_ratio=args.max_edge_ratio,
            confidence_threshold=args.confidence_threshold,
            lambda_c=lc,
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

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate_model(model, data, test_edges, val_edges, eval_edge_index_dict,
                   test_pos, val_pos, train_pos, device):
    _, test_auc, test_auprc = test(data, model, test_edges, eval_edge_index_dict)
    rank = evaluate_ranking(
        data, model, eval_edge_index_dict,
        test_pos, train_pos, val_pos, test_pos)
    return {
        'test_auc': float(test_auc),
        'test_auprc': float(test_auprc),
        'test_mrr': float(rank['MRR']),
        'test_h3': float(rank['Hits@3']),
        'test_h10': float(rank['Hits@10']),
    }


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Seeds: {args.seeds}')
    print(f'Fractions: {args.fractions}')

    data_path = osp.join('data', 'hetionet_data.pt')
    try:
        data_raw = torch.load(data_path, weights_only=False)
    except TypeError:
        data_raw = torch.load(data_path)

    results = {}  # (fraction, model_type, seed) -> metrics

    for seed in args.seeds:
        print(f'\n{"=" * 60}\n[Seed {seed}]\n{"=" * 60}')
        set_seed(seed)

        # Split (seed 별로 같은 val/test 사용)
        transform = RandomLinkSplit(
            num_val=0.1, num_test=0.1,
            is_undirected=True,
            add_negative_train_samples=False,
            neg_sampling_ratio=1.0,
            edge_types=[EDGE_TYPE],
            rev_edge_types=[REV_EDGE_TYPE],
        )
        train_data_full, val_data, test_data = transform(data_raw)

        def _split(sd):
            return {'index': sd[EDGE_TYPE].edge_label_index,
                    'label': sd[EDGE_TYPE].edge_label}
        val_edges = _split(val_data)
        test_edges = _split(test_data)
        pos_mask_v = val_edges['label'] > 0.5
        pos_mask_t = test_edges['label'] > 0.5

        data = data_raw.to(device)
        train_data_full = train_data_full.to(device)
        for s in (val_edges, test_edges):
            s['index'] = s['index'].to(device)
            s['label'] = s['label'].to(device)

        val_pos = val_edges['index'][:, pos_mask_v]
        test_pos = test_edges['index'][:, pos_mask_t]

        for frac in args.fractions:
            print(f'\n--- fraction={frac:.2f} ---')
            # Subsample
            train_data_sub, n_keep = subsample_treats(train_data_full, frac, seed)
            train_data_sub = train_data_sub.to(device)
            train_pos = train_data_sub[EDGE_TYPE].edge_index
            eval_edge_index_dict = {k: v for k, v in train_data_sub.edge_index_dict.items()}

            print(f'  Train treats edges: {n_keep}')

            # HeteroPULL
            set_seed(seed)
            t0 = time.time()
            m_ours = train_model(args, data, train_data_sub, val_edges,
                                  eval_edge_index_dict, device, use_pu=True)
            eval_ours = evaluate_model(m_ours, data, test_edges, val_edges,
                                        eval_edge_index_dict,
                                        test_pos, val_pos, train_pos, device)
            print(f'  HeteroPULL  AUC={eval_ours["test_auc"]:.4f} '
                  f'AUPRC={eval_ours["test_auprc"]:.4f} '
                  f'MRR={eval_ours["test_mrr"]:.4f} '
                  f'H@10={eval_ours["test_h10"]:.3f} '
                  f'({time.time()-t0:.1f}s)')

            # HGT baseline
            set_seed(seed)
            t0 = time.time()
            m_hgt = train_model(args, data, train_data_sub, val_edges,
                                 eval_edge_index_dict, device, use_pu=False)
            eval_hgt = evaluate_model(m_hgt, data, test_edges, val_edges,
                                       eval_edge_index_dict,
                                       test_pos, val_pos, train_pos, device)
            print(f'  HGT         AUC={eval_hgt["test_auc"]:.4f} '
                  f'AUPRC={eval_hgt["test_auprc"]:.4f} '
                  f'MRR={eval_hgt["test_mrr"]:.4f} '
                  f'H@10={eval_hgt["test_h10"]:.3f} '
                  f'({time.time()-t0:.1f}s)')

            key = f'{frac:.2f}'
            if key not in results:
                results[key] = {'HeteroPULL': [], 'HGT': [], 'n_train_edges': n_keep}
            results[key]['HeteroPULL'].append({'seed': seed, **eval_ours})
            results[key]['HGT'].append({'seed': seed, **eval_hgt})

    # Aggregate
    summary = {}
    for frac_key, d in results.items():
        summary[frac_key] = {'n_train_edges': d['n_train_edges']}
        for m in ('HeteroPULL', 'HGT'):
            runs = d[m]
            agg = {}
            for metric in ('test_auc', 'test_auprc', 'test_mrr', 'test_h3', 'test_h10'):
                vals = [r[metric] for r in runs]
                mean = sum(vals) / len(vals)
                std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) \
                    if len(vals) > 1 else 0
                agg[metric] = {'mean': round(mean, 4), 'std': round(std, 4)}
            summary[frac_key][m] = agg

    # Save
    out_dir = osp.dirname(args.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = {
        'seeds': args.seeds,
        'fractions': args.fractions,
        'summary': summary,
        'raw_runs': results,
    }
    with open(args.out_json, 'w') as f:
        json.dump(payload, f, indent=2)

    # Print final summary table
    print(f'\n\n{"=" * 70}\n[최종 요약 — mean±std]\n{"=" * 70}')
    print(f'{"Fraction":>10} {"Model":>12} {"AUPRC":>14} {"MRR":>14} {"H@10":>14}')
    print('-' * 70)
    for frac_key in sorted(summary.keys()):
        for m in ('HeteroPULL', 'HGT'):
            a = summary[frac_key][m]
            print(f'{frac_key:>10} {m:>12} '
                  f'{a["test_auprc"]["mean"]:.3f}±{a["test_auprc"]["std"]:.3f}   '
                  f'{a["test_mrr"]["mean"]:.3f}±{a["test_mrr"]["std"]:.3f}   '
                  f'{a["test_h10"]["mean"]:.3f}±{a["test_h10"]["std"]:.3f}')

    print(f'\n[저장] {args.out_json}')


if __name__ == '__main__':
    main()
