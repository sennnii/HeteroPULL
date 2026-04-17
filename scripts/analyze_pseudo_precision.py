"""
Epoch별 pseudo-positive precision 분석.

expand_graph가 추가하는 pseudo-positive 엣지 중에서
val/test positive에 실제로 존재하는 비율(precision)을 epoch마다 측정.

이를 통해 "초기 잘못된 pseudo-positive가 후반에 악영향을 주는가?"에 답변.

사용 예시:
    CUDA_VISIBLE_DEVICES=1 python scripts/analyze_pseudo_precision.py \
        --seed 0 --epochs 50 --patience 50 \
        --hidden_dim 128 --out_dim 64 --heads 4 --layers 2 \
        --confidence_threshold 0.90 --growth_rate 0.03 \
        --out_json results/pseudo_precision/seed0.json
"""

import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
from torch_geometric.transforms import RandomLinkSplit

from src.model_hetionet import HeteroPULLModel
from src.train_hetionet import (
    train, test, evaluate_ranking, expand_graph,
    EDGE_TYPE, REV_EDGE_TYPE,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--out_dim', type=int, default=64)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=50,
                        help="precision 분석은 전 epoch를 봐야 하므로 patience를 크게")
    parser.add_argument('--inner_steps', type=int, default=50)
    parser.add_argument('--growth_rate', type=float, default=0.03)
    parser.add_argument('--max_edge_ratio', type=float, default=1.0)
    parser.add_argument('--confidence_threshold', type=float, default=0.85)
    parser.add_argument('--lambda_c', type=float, default=1.0)
    parser.add_argument('--no_morgan', action='store_true')
    parser.add_argument('--out_json', type=str, default='results/pseudo_precision/analysis.json')
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def compute_pseudo_precision(expanded_edge, val_pos_set, test_pos_set):
    """
    pseudo-positive 엣지 중 val/test positive에 포함되는 비율 계산.

    PU learning에서 train에 없지만 실제 positive인 엣지 = val/test positive.
    pseudo-positive가 이들을 '재발견'하면 모델이 올바른 방향으로 확장 중인 것.
    """
    if expanded_edge.size(1) == 0:
        return {'n_pseudo': 0, 'n_hit_val': 0, 'n_hit_test': 0,
                'n_hit_any': 0, 'precision_val': 0.0, 'precision_test': 0.0,
                'precision_any': 0.0, 'mean_confidence': 0.0}

    n_pseudo = expanded_edge.size(1)
    pseudo_set = set()
    for i in range(n_pseudo):
        pseudo_set.add((expanded_edge[0, i].item(), expanded_edge[1, i].item()))

    n_hit_val = len(pseudo_set & val_pos_set)
    n_hit_test = len(pseudo_set & test_pos_set)
    n_hit_any = len(pseudo_set & (val_pos_set | test_pos_set))

    return {
        'n_pseudo': n_pseudo,
        'n_hit_val': n_hit_val,
        'n_hit_test': n_hit_test,
        'n_hit_any': n_hit_any,
        'precision_val': n_hit_val / n_pseudo,
        'precision_test': n_hit_test / n_pseudo,
        'precision_any': n_hit_any / n_pseudo,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, Seed: {args.seed}")

    data = torch.load('data/hetionet_data.pt', weights_only=False)

    transform = RandomLinkSplit(
        num_val=0.1, num_test=0.1, is_undirected=True,
        add_negative_train_samples=False, neg_sampling_ratio=1.0,
        edge_types=[EDGE_TYPE], rev_edge_types=[REV_EDGE_TYPE],
    )
    train_data, val_data, test_data = transform(data)

    # val/test positive edge set 구축 (precision 측정용 ground truth)
    val_label_idx = val_data[EDGE_TYPE].edge_label_index
    val_label = val_data[EDGE_TYPE].edge_label
    val_pos_idx = val_label_idx[:, val_label > 0.5]
    val_pos_set = set()
    for i in range(val_pos_idx.size(1)):
        val_pos_set.add((val_pos_idx[0, i].item(), val_pos_idx[1, i].item()))

    test_label_idx = test_data[EDGE_TYPE].edge_label_index
    test_label = test_data[EDGE_TYPE].edge_label
    test_pos_idx = test_label_idx[:, test_label > 0.5]
    test_pos_set = set()
    for i in range(test_pos_idx.size(1)):
        test_pos_set.add((test_pos_idx[0, i].item(), test_pos_idx[1, i].item()))

    print(f"|val_pos|={len(val_pos_set)}, |test_pos|={len(test_pos_set)}")

    data = data.to(device)
    train_data = train_data.to(device)

    model = HeteroPULLModel(
        data=data,
        hidden_channels=args.hidden_dim,
        out_channels=args.out_dim,
        num_heads=args.heads,
        num_layers=args.layers,
        use_compound_features=(not args.no_morgan),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    orig_edge = train_data[EDGE_TYPE].edge_index
    eval_edge_index_dict = {k: v for k, v in train_data.edge_index_dict.items()}

    # Ranking eval용
    train_pos_all = train_data[EDGE_TYPE].edge_index
    val_pos_only = val_pos_idx.to(device)
    test_pos_only = test_pos_idx.to(device)

    val_edges = {
        'index': val_label_idx.to(device),
        'label': val_label.to(device),
    }
    test_edges = {
        'index': test_label_idx.to(device),
        'label': test_label.to(device),
    }

    epoch_records = []
    z_dict_prev = None

    print(f"\n학습 시작 (epochs={args.epochs}, r={args.growth_rate}, "
          f"tau={args.confidence_threshold}, lambda_c={args.lambda_c})")

    for epoch in range(1, args.epochs + 1):
        # 1) expand_graph를 직접 호출하여 pseudo-positive precision 측정
        if epoch > 1 and z_dict_prev is not None:
            expanded_edge, expanded_weight = expand_graph(
                model, z_dict_prev, orig_edge, epoch,
                growth_rate=args.growth_rate,
                max_edge_ratio=args.max_edge_ratio,
                confidence_threshold=args.confidence_threshold,
            )
            prec_info = compute_pseudo_precision(
                expanded_edge, val_pos_set, test_pos_set
            )
            prec_info['mean_confidence'] = float(expanded_weight.mean().item()) if expanded_weight.numel() > 0 else 0.0
            prec_info['min_confidence'] = float(expanded_weight.min().item()) if expanded_weight.numel() > 0 else 0.0
        else:
            prec_info = {
                'n_pseudo': 0, 'n_hit_val': 0, 'n_hit_test': 0,
                'n_hit_any': 0, 'precision_val': 0.0, 'precision_test': 0.0,
                'precision_any': 0.0, 'mean_confidence': 0.0, 'min_confidence': 0.0,
            }

        # 2) 실제 학습
        loss, z_dict_prev, n_exp = train(
            model, optimizer, data, train_data, epoch,
            z_dict_prev=z_dict_prev,
            inner_steps=args.inner_steps,
            growth_rate=args.growth_rate,
            max_edge_ratio=args.max_edge_ratio,
            confidence_threshold=args.confidence_threshold,
            lambda_c=args.lambda_c,
        )

        # 3) 평가
        _, val_auc, val_auprc = test(data, model, val_edges, eval_edge_index_dict)
        _, test_auc, test_auprc = test(data, model, test_edges, eval_edge_index_dict)
        val_rank = evaluate_ranking(
            data, model, eval_edge_index_dict,
            val_pos_only, train_pos_all, val_pos_only, test_pos_only,
        )
        test_rank = evaluate_ranking(
            data, model, eval_edge_index_dict,
            test_pos_only, train_pos_all, val_pos_only, test_pos_only,
        )

        record = {
            'epoch': epoch,
            'loss': float(loss),
            'n_exp': n_exp,
            **{f'pseudo_{k}': v for k, v in prec_info.items()},
            'val_auc': float(val_auc),
            'val_auprc': float(val_auprc),
            'val_mrr': float(val_rank['MRR']),
            'val_h10': float(val_rank['Hits@10']),
            'test_auc': float(test_auc),
            'test_auprc': float(test_auprc),
            'test_mrr': float(test_rank['MRR']),
            'test_h10': float(test_rank['Hits@10']),
        }
        epoch_records.append(record)

        print(f"Epoch {epoch:02d} | Loss={loss:.4f} | "
              f"|E_exp|={n_exp} | "
              f"pseudo_prec(any)={prec_info['precision_any']:.3f} "
              f"({prec_info['n_hit_any']}/{prec_info['n_pseudo']}) | "
              f"conf={prec_info['mean_confidence']:.3f} | "
              f"Val MRR={val_rank['MRR']:.4f} AUC={val_auc:.4f} | "
              f"Test MRR={test_rank['MRR']:.4f} AUC={test_auc:.4f}")

    # 저장
    out_dir = os.path.dirname(args.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    result = {
        'args': vars(args),
        'val_pos_count': len(val_pos_set),
        'test_pos_count': len(test_pos_set),
        'epoch_records': epoch_records,
    }
    with open(args.out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n결과 저장: {args.out_json}")


if __name__ == '__main__':
    main()
