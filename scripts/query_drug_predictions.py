"""
질병 쿼리 → top-K 약물 재창출 후보 예측.

학습된 HeteroPULL 모델로 특정 질병에 대한 약물 후보를 랭킹.
이미 알려진 치료 관계(train/val/test positive)는 제외하고 신규 예측만 표시.

사용 예시:
    # 미리 정의된 대표 질병들에 대해 각각 top-5 약물 출력
    CUDA_VISIBLE_DEVICES=1 python scripts/query_drug_predictions.py \
        --seed 0 --epochs 50 --patience 15 \
        --hidden_dim 128 --out_dim 64 --heads 4 --layers 2 \
        --confidence_threshold 0.90 --growth_rate 0.03 --lambda_c 1.0 \
        --top_k 5 \
        --query_diseases "Alzheimer's disease" "Parkinson's disease" "breast cancer" \
                        "type 2 diabetes mellitus" "Crohn's disease" "schizophrenia" \
        --out_json results/drug_queries/queries.json
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
    train, test, evaluate_ranking,
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
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--inner_steps', type=int, default=50)
    parser.add_argument('--growth_rate', type=float, default=0.03)
    parser.add_argument('--max_edge_ratio', type=float, default=1.0)
    parser.add_argument('--confidence_threshold', type=float, default=0.90)
    parser.add_argument('--lambda_c', type=float, default=1.0)
    parser.add_argument('--no_morgan', action='store_true')
    parser.add_argument('--top_k', type=int, default=5,
                        help="질병당 출력할 약물 후보 수")
    parser.add_argument('--query_diseases', type=str, nargs='+',
                        default=[
                            "Alzheimer's disease",
                            "Parkinson's disease",
                            "breast cancer",
                            "type 2 diabetes mellitus",
                            "Crohn's disease",
                            "schizophrenia",
                        ])
    parser.add_argument('--out_json', type=str,
                        default='results/drug_queries/queries.json')
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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

    # 이미 알려진 관계 저장 (쿼리 시 제외용)
    def _pos_only(split_data):
        idx = split_data[EDGE_TYPE].edge_label_index
        lab = split_data[EDGE_TYPE].edge_label
        return idx[:, lab > 0.5]

    train_pos = train_data[EDGE_TYPE].edge_index
    val_pos = _pos_only(val_data)
    test_pos = _pos_only(test_data)

    data[EDGE_TYPE]['train_pos_edge_index'] = train_pos
    data[EDGE_TYPE]['val_pos_edge_index'] = val_pos
    data[EDGE_TYPE]['test_pos_edge_index'] = test_pos

    data = data.to(device)
    train_data = train_data.to(device)

    # Val/test edge 집합 for early stopping
    def _split_edges(split_data):
        return {
            'index': split_data[EDGE_TYPE].edge_label_index.to(device),
            'label': split_data[EDGE_TYPE].edge_label.to(device),
        }
    val_edges = _split_edges(val_data)
    test_edges = _split_edges(test_data)
    val_pos_only = val_pos.to(device)
    test_pos_only = test_pos.to(device)
    train_pos_all = train_pos.to(device)

    model = HeteroPULLModel(
        data=data, hidden_channels=args.hidden_dim, out_channels=args.out_dim,
        num_heads=args.heads, num_layers=args.layers,
        use_compound_features=(not args.no_morgan),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    eval_edge_index_dict = {k: v for k, v in train_data.edge_index_dict.items()}

    print(f"\n학습 시작 (epochs={args.epochs})")
    best_val_mrr = -1.0
    best_state_dict = None
    best_epoch = 0
    patience_counter = 0
    z_dict_prev = None

    for epoch in range(1, args.epochs + 1):
        loss, z_dict_prev, n_exp = train(
            model, optimizer, data, train_data, epoch, z_dict_prev=z_dict_prev,
            inner_steps=args.inner_steps, growth_rate=args.growth_rate,
            max_edge_ratio=args.max_edge_ratio,
            confidence_threshold=args.confidence_threshold,
            lambda_c=args.lambda_c,
        )
        val_rank = evaluate_ranking(
            data, model, eval_edge_index_dict,
            val_pos_only, train_pos_all, val_pos_only, test_pos_only,
        )
        if val_rank['MRR'] > best_val_mrr:
            best_val_mrr = val_rank['MRR']
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stop @ epoch {epoch}")
                break
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:02d} | Loss={loss:.4f} | Val MRR={val_rank['MRR']:.4f}")

    print(f"\nBest epoch: {best_epoch} | Val MRR: {best_val_mrr:.4f}")
    model.load_state_dict(best_state_dict)

    # ===== 쿼리 수행 =====
    model.eval()
    with torch.no_grad():
        z_dict = model.encode(data, eval_edge_index_dict, None)
        raw_scores = model.score_all(z_dict).cpu()  # [n_compound, n_disease]

        # 알려진 관계는 -inf로 마스킹
        for key_name in ['train_pos_edge_index', 'val_pos_edge_index', 'test_pos_edge_index']:
            if key_name in data[EDGE_TYPE]:
                ei = data[EDGE_TYPE][key_name].cpu()
                raw_scores[ei[0], ei[1]] = -float('inf')

        probs = torch.sigmoid(raw_scores)

    compound_names = data.node_names['Compound']  # dict: idx -> name
    disease_names = data.node_names['Disease']    # dict: idx -> name

    # disease name -> idx 매핑
    name_to_idx = {}
    for idx, name in disease_names.items():
        name_to_idx[name.lower()] = idx

    results = {}
    print(f"\n{'=' * 70}")
    print(f"질병 쿼리 → Top-{args.top_k} 약물 후보")
    print('=' * 70)

    for q in args.query_diseases:
        d_idx = name_to_idx.get(q.lower())
        if d_idx is None:
            # 부분 매칭
            candidates = [(idx, n) for idx, n in disease_names.items() if q.lower() in n.lower()]
            if candidates:
                d_idx, q_full = candidates[0]
                print(f"\n[Query] '{q}' → '{q_full}' (idx={d_idx})")
                q = q_full
            else:
                print(f"\n[Query] '{q}' → 찾을 수 없음. 스킵.")
                continue
        else:
            print(f"\n[Query] {q} (idx={d_idx})")

        col = probs[:, d_idx]
        top_vals, top_idx = torch.topk(col, args.top_k)

        drugs = []
        for rank, (p, c_idx) in enumerate(zip(top_vals.tolist(), top_idx.tolist()), 1):
            c_name = compound_names[c_idx]
            print(f"  {rank}. {c_name:<40s}  (prob={p:.4f})")
            drugs.append({'rank': rank, 'compound': c_name, 'prob': float(p)})

        results[q] = {'disease_idx': int(d_idx), 'top_drugs': drugs}

    # 저장
    out_dir = os.path.dirname(args.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump({
            'args': vars(args),
            'best_epoch': best_epoch,
            'best_val_mrr': float(best_val_mrr),
            'queries': results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {args.out_json}")


if __name__ == '__main__':
    main()
