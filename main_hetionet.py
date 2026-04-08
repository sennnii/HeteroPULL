import os.path as osp
import argparse
import random
import time
import numpy as np
import torch
from torch_geometric.transforms import RandomLinkSplit

from src.model_hetionet import HeteroPULLModel
from src.train_hetionet import train, test, get_drug_repurposing_candidates


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
    parser.add_argument('--prior', type=float, default=None,
                        help="nnPU class prior π_p. None이면 |treats|/(|C|·|D|)로 자동 추정. "
                             "ablation 시 {1e-3, 3e-3, 1e-2, 3e-2} 권장")
    parser.add_argument('--unl_ratio', type=int, default=5,
                        help="nnPU unlabeled batch = positive의 k배")
    parser.add_argument('--inner_steps', type=int, default=50)
    parser.add_argument('--verbose', type=str, default="y")
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
        # 구 PyTorch 호환 (weights_only 인자 미지원)
        data = torch.load(data_path)

    print("데이터 분할 중 (Train/Val/Test)...")
    edge_type_to_predict = ('Compound', 'treats', 'Disease')
    rev_edge_type_to_predict = ('Disease', 'rev_treats', 'Compound')

    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        disjoint_train_ratio=0.3,
        add_negative_train_samples=False,
        neg_sampling_ratio=1.0,
        edge_types=[edge_type_to_predict],
        rev_edge_types=[rev_edge_type_to_predict],
    )

    train_data, val_data, test_data = transform(data)

    # RandomLinkSplit(neg_sampling_ratio=1.0)은 val/test의 edge_label_index에
    # positive + negative를 모두 담고, edge_label(1/0)로 구분한다.
    def _split_edges(split_data):
        return {
            'index': split_data[edge_type_to_predict].edge_label_index,
            'label': split_data[edge_type_to_predict].edge_label,
        }

    val_edges = _split_edges(val_data)
    test_edges = _split_edges(test_data)

    # 후보 발굴 시 이미 사용된 positive 엣지를 제외하기 위한 인덱스 저장
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
    print(f"학습용 'treats' MP 엣지: {train_data[edge_type_to_predict].edge_index.shape[1]}")
    print(f"학습용 'treats' Supervision 엣지: {int((train_data[edge_type_to_predict].edge_label > 0.5).sum())}")
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

    # 평가 시 encoder 입력: train MP(70%) + supervision(30%) 전체 = 전체 training treats.
    # MP partition 만 쓰면 학습된 정보의 30%가 eval 시점에 인코더에서 사라져 AUC가 손해.
    # val/test positive는 여전히 포함되지 않으므로 leakage는 없다.
    tr_mp = train_data[edge_type_to_predict].edge_index
    tr_sup_mask = train_data[edge_type_to_predict].edge_label > 0.5
    tr_sup = train_data[edge_type_to_predict].edge_label_index[:, tr_sup_mask]
    tr_full = torch.cat([tr_mp, tr_sup], dim=1)
    eval_edge_index_dict = {k: v for k, v in train_data.edge_index_dict.items()}
    eval_edge_index_dict[edge_type_to_predict] = tr_full
    if rev_edge_type_to_predict in eval_edge_index_dict:
        eval_edge_index_dict[rev_edge_type_to_predict] = tr_full.flip(0)

    # nnPU class prior
    if args.prior is None:
        prior_used = None  # train() 내부에서 자동 추정
    else:
        prior_used = args.prior
        print(f"nnPU prior π_p 수동 지정: {prior_used}")

    print("\n[HeteroPULL 학습 시작]")
    best_val_auc = 0
    best_test_auc = 0
    best_epoch = 0
    patience_counter = 0

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        loss, _ = train(model, optimizer, data, train_data, epoch,
                        inner_steps=args.inner_steps,
                        unl_ratio=args.unl_ratio,
                        prior=prior_used)
        val_loss, val_auc = test(data, model, val_edges, eval_edge_index_dict)
        curr_test_loss, curr_test_auc = test(data, model, test_edges, eval_edge_index_dict)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_test_auc = curr_test_auc
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early Stopping: {args.patience} Epoch 동안 성능 향상이 없어 조기 종료합니다.")
                break

        if args.verbose == 'y':
            epoch_time = time.time() - epoch_start_time
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, '
                  f'Test AUC: {curr_test_auc:.4f} (Patience: {patience_counter}/{args.patience}, Time: {epoch_time:.2f}s)')

    print("\n[학습 완료]")
    print(f'Best Epoch: {best_epoch:02d}, Val AUC: {best_val_auc:.4f}, Best Test AUC: {best_test_auc:.4f}')
    print(f'총 학습 시간: {(time.time() - start_time):.2f}s')

    get_drug_repurposing_candidates(data, model, eval_edge_index_dict, num_candidates=20)


if __name__ == '__main__':
    main()
