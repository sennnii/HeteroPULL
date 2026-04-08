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
from src.train_hetionet import train, test, evaluate_ranking, get_drug_repurposing_candidates


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--out_dim', type=int, default=64)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--inner_steps', type=int, default=50)
    # PULL hyperparameters
    parser.add_argument('--growth_rate', type=float, default=0.03,
                        help="epoch당 확장 비율 r. |E_exp|=r·|E_orig|·(epoch-1)")
    parser.add_argument('--max_edge_ratio', type=float, default=1.0,
                        help="원본 대비 최대 확장 비율 cap")
    parser.add_argument('--confidence_threshold', type=float, default=0.85,
                        help="sigmoid(score) 이 값 이상인 쌍만 pseudo-positive 로 인정")
    parser.add_argument('--lambda_c', type=float, default=1.0,
                        help="Correction loss 가중치: L = L_E + λ_c · L_C")
    parser.add_argument('--verbose', type=str, default="y")
    parser.add_argument('--result_json', type=str, default=None,
                        help="지정 시 final metric 을 JSON 파일로 저장 (seed sweep 집계용)")
    parser.add_argument('--skip_candidates', action='store_true',
                        help="약물 재창출 후보 top-K 출력 생략 (seed sweep 시 로그 단축용)")
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
    print(f"사용 장치: {device}")

    data_path = osp.join('data', 'hetionet_data.pt')
    if not osp.exists(data_path):
        print(f"오류: {data_path} 파일을 찾을 수 없습니다.")
        print("먼저 `python preprocess_hetionet.py`를 실행하여 데이터를 전처리하세요.")
        return

    print("전처리된 Hetionet 데이터 로드 중...")
    try:
        data = torch.load(data_path, weights_only=False)
    except TypeError:
        data = torch.load(data_path)

    print("데이터 분할 중 (Train/Val/Test)...")
    edge_type_to_predict = ('Compound', 'treats', 'Disease')
    rev_edge_type_to_predict = ('Disease', 'rev_treats', 'Compound')

    # PULL 구조에서는 train edge 전체가 MP 이자 supervision (expected graph).
    # 따라서 disjoint_train_ratio 는 사용하지 않는다.
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=1.0,
        edge_types=[edge_type_to_predict],
        rev_edge_types=[rev_edge_type_to_predict],
    )

    train_data, val_data, test_data = transform(data)

    # RandomLinkSplit 은 val/test 의 edge_label_index 에 pos+neg 를 함께 저장하고
    # edge_label(1/0) 로 구분한다.
    def _split_edges(split_data):
        return {
            'index': split_data[edge_type_to_predict].edge_label_index,
            'label': split_data[edge_type_to_predict].edge_label,
        }

    val_edges = _split_edges(val_data)
    test_edges = _split_edges(test_data)

    # 후보 발굴 시 이미 사용된 positive 를 마스킹하기 위한 인덱스 저장
    pos_mask_val = val_edges['label'] > 0.5
    pos_mask_test = test_edges['label'] > 0.5
    data[edge_type_to_predict]['train_pos_edge_index'] = train_data[edge_type_to_predict].edge_index
    data[edge_type_to_predict]['val_pos_edge_index'] = val_edges['index'][:, pos_mask_val]
    data[edge_type_to_predict]['test_pos_edge_index'] = test_edges['index'][:, pos_mask_test]

    data = data.to(device)
    train_data = train_data.to(device)
    for split in (val_edges, test_edges):
        split['index'] = split['index'].to(device)
        split['label'] = split['label'].to(device)

    n_val_pos = int(pos_mask_val.sum())
    n_test_pos = int(pos_mask_test.sum())
    print("\n[데이터 로드 완료]")
    print(f"학습용 'treats' 엣지: {train_data[edge_type_to_predict].edge_index.shape[1]}")
    print(f"검증용 'treats' 엣지 (Pos/Neg): {n_val_pos}/{val_edges['label'].numel() - n_val_pos}")
    print(f"테스트용 'treats' 엣지 (Pos/Neg): {n_test_pos}/{test_edges['label'].numel() - n_test_pos}")

    model = HeteroPULLModel(
        data=data,
        hidden_channels=args.hidden_dim,
        out_channels=args.out_dim,
        num_heads=args.heads,
        num_layers=args.layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 평가 시 encoder 입력: train treats 엣지만 사용 (val/test 엣지 미노출).
    # train_data.edge_index_dict 가 정확히 이 조건을 만족한다.
    eval_edge_index_dict = {k: v for k, v in train_data.edge_index_dict.items()}

    # Ranking 평가에 쓸 positive 엣지들 (filtered setting 용)
    train_pos_all = train_data[edge_type_to_predict].edge_index
    val_pos_only = val_edges['index'][:, pos_mask_val]
    test_pos_only = test_edges['index'][:, pos_mask_test]

    def _full_eval(tag):
        """AUC/AUPRC + MRR/Hits@k 를 val/test 둘 다에 대해 계산해서 한 줄로 출력."""
        _, v_auc, v_auprc = test(data, model, val_edges, eval_edge_index_dict)
        _, t_auc, t_auprc = test(data, model, test_edges, eval_edge_index_dict)
        v_rank = evaluate_ranking(
            data, model, eval_edge_index_dict,
            val_pos_only, train_pos_all, val_pos_only, test_pos_only,
        )
        t_rank = evaluate_ranking(
            data, model, eval_edge_index_dict,
            test_pos_only, train_pos_all, val_pos_only, test_pos_only,
        )
        print(f"[{tag}] "
              f"Val  AUC={v_auc:.4f} AUPRC={v_auprc:.4f} "
              f"MRR={v_rank['MRR']:.4f} H@1={v_rank['Hits@1']:.3f} "
              f"H@3={v_rank['Hits@3']:.3f} H@10={v_rank['Hits@10']:.3f}")
        print(f"[{tag}] "
              f"Test AUC={t_auc:.4f} AUPRC={t_auprc:.4f} "
              f"MRR={t_rank['MRR']:.4f} H@1={t_rank['Hits@1']:.3f} "
              f"H@3={t_rank['Hits@3']:.3f} H@10={t_rank['Hits@10']:.3f}")
        return {
            'val_auc': float(v_auc), 'val_auprc': float(v_auprc),
            'val_mrr': float(v_rank['MRR']),
            'val_h1': float(v_rank['Hits@1']),
            'val_h3': float(v_rank['Hits@3']),
            'val_h10': float(v_rank['Hits@10']),
            'test_auc': float(t_auc), 'test_auprc': float(t_auprc),
            'test_mrr': float(t_rank['MRR']),
            'test_h1': float(t_rank['Hits@1']),
            'test_h3': float(t_rank['Hits@3']),
            'test_h10': float(t_rank['Hits@10']),
        }

    # ---------- Baseline: random init 평가 (학습 전) ----------
    print("\n[Epoch 00 — Baseline: Random Init]")
    _full_eval("Ep00")

    print("\n[HeteroPULL 학습 시작]")
    print("Early stopping 기준: Val MRR (filtered)")
    best_val_mrr = -1.0
    best_val_auc = 0.0
    best_test_auc = 0.0
    best_test_mrr = 0.0
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0
    z_dict_prev = None

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        loss, z_dict_prev, n_exp = train(
            model, optimizer, data, train_data, epoch,
            z_dict_prev=z_dict_prev,
            inner_steps=args.inner_steps,
            growth_rate=args.growth_rate,
            max_edge_ratio=args.max_edge_ratio,
            confidence_threshold=args.confidence_threshold,
            lambda_c=args.lambda_c,
        )
        val_loss, val_auc, val_auprc = test(data, model, val_edges, eval_edge_index_dict)
        curr_test_loss, curr_test_auc, curr_test_auprc = test(data, model, test_edges, eval_edge_index_dict)

        # Ranking metrics (filtered setting)
        val_rank = evaluate_ranking(
            data, model, eval_edge_index_dict,
            val_pos_only, train_pos_all, val_pos_only, test_pos_only,
        )
        test_rank = evaluate_ranking(
            data, model, eval_edge_index_dict,
            test_pos_only, train_pos_all, val_pos_only, test_pos_only,
        )

        curr_val_mrr = val_rank['MRR']
        if curr_val_mrr > best_val_mrr:
            best_val_mrr = curr_val_mrr
            best_test_mrr = test_rank['MRR']
            best_val_auc = val_auc
            best_test_auc = curr_test_auc
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early Stopping: {args.patience} Epoch 동안 성능 향상이 없어 조기 종료합니다.")
                break

        if args.verbose == 'y':
            epoch_time = time.time() - epoch_start_time
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, |E_exp|: {n_exp}, '
                  f'Val AUC: {val_auc:.4f} AUPRC: {val_auprc:.4f} '
                  f'MRR: {val_rank["MRR"]:.4f} H@10: {val_rank["Hits@10"]:.3f}, '
                  f'Test AUC: {curr_test_auc:.4f} MRR: {test_rank["MRR"]:.4f} '
                  f'(Patience: {patience_counter}/{args.patience}, Time: {epoch_time:.2f}s)')

    print("\n[학습 완료]")
    print(f'Best Epoch (by Val MRR): {best_epoch:02d}')
    print(f'  Val  AUC: {best_val_auc:.4f}  MRR: {best_val_mrr:.4f}')
    print(f'  Test AUC: {best_test_auc:.4f}  MRR: {best_test_mrr:.4f}')
    print(f'총 학습 시간: {(time.time() - start_time):.2f}s')

    # Best checkpoint 복원 후 최종 평가
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"\n[Best checkpoint 복원 완료: epoch {best_epoch:02d}]")
    print()
    final_metrics = _full_eval("Final")

    if args.result_json is not None:
        out_dir = osp.dirname(args.result_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = {
            'seed': args.seed,
            'best_epoch': best_epoch,
            'args': {k: v for k, v in vars(args).items() if k != 'result_json'},
            'final': final_metrics,
        }
        with open(args.result_json, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"[result_json 저장] {args.result_json}")

    if not args.skip_candidates:
        get_drug_repurposing_candidates(data, model, eval_edge_index_dict, num_candidates=20)


if __name__ == '__main__':
    main()
