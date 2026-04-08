import os.path as osp
import argparse
import random
import time
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling

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

    data[edge_type_to_predict]['train_pos_edge_index'] = train_data[edge_type_to_predict].edge_index
    data[edge_type_to_predict]['val_pos_edge_index'] = val_data[edge_type_to_predict].edge_label_index
    data[edge_type_to_predict]['test_pos_edge_index'] = test_data[edge_type_to_predict].edge_label_index

    val_edges = {
        'pos': val_data[edge_type_to_predict].edge_label_index,
        'neg': val_data[edge_type_to_predict].get(
            'neg_edge_label_index', torch.empty((2, 0), dtype=torch.long)),
    }
    test_edges = {
        'pos': test_data[edge_type_to_predict].edge_label_index,
        'neg': test_data[edge_type_to_predict].get(
            'neg_edge_label_index', torch.empty((2, 0), dtype=torch.long)),
    }

    for name, split in [('Validation', val_edges), ('Test', test_edges)]:
        if split['neg'].shape[1] == 0:
            print(f"  - {name} negative 샘플 생성 중...")
            split['neg'] = negative_sampling(
                edge_index=train_data[edge_type_to_predict].edge_index,
                num_nodes=(data['Compound'].num_nodes, data['Disease'].num_nodes),
                num_neg_samples=split['pos'].size(1),
                method='sparse',
            )

    data = data.to(device)
    train_data = train_data.to(device)
    for split in (val_edges, test_edges):
        split['pos'] = split['pos'].to(device)
        split['neg'] = split['neg'].to(device)

    print("\n[데이터 로드 완료]")
    print(f"학습용 'treats' 엣지: {train_data[edge_type_to_predict].edge_index.shape[1]}")
    print(f"검증용 'treats' 엣지 (Pos/Neg): {val_edges['pos'].shape[1]}/{val_edges['neg'].shape[1]}")
    print(f"테스트용 'treats' 엣지 (Pos/Neg): {test_edges['pos'].shape[1]}/{test_edges['neg'].shape[1]}")

    model = HeteroPULLModel(
        data=data,
        hidden_channels=args.hidden_dim,
        out_channels=args.out_dim,
        num_heads=args.heads,
        num_layers=args.layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = BCEWithLogitsLoss()

    print("\n[HeteroPULL 학습 시작]")
    best_val_auc = 0
    best_test_auc = 0
    best_epoch = 0
    z_dict = None
    patience_counter = 0

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        loss, z_dict, _, _ = train(model, optimizer, data, train_data, criterion, epoch, z_dict)
        val_loss, val_auc = test(data, model, val_edges, criterion)
        curr_test_loss, curr_test_auc = test(data, model, test_edges, criterion)

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

    get_drug_repurposing_candidates(data, model, num_candidates=20)


if __name__ == '__main__':
    main()
