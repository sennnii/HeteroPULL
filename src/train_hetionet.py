from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling


EDGE_TYPE = ('Compound', 'treats', 'Disease')
REV_EDGE_TYPE = ('Disease', 'rev_treats', 'Compound')


def _build_edge_index_dict(train_data, treats_edge_index):
    """treats 와 rev_treats 를 주어진 edge_index 로 교체한 dict 반환."""
    d = {k: v for k, v in train_data.edge_index_dict.items()}
    d[EDGE_TYPE] = treats_edge_index
    if REV_EDGE_TYPE in d:
        d[REV_EDGE_TYPE] = treats_edge_index.flip(0)
    return d


@torch.no_grad()
def expand_graph(model, z_dict, orig_edge_index, epoch,
                 growth_rate=0.03, max_edge_ratio=1.0,
                 confidence_threshold=0.85):
    """
    PULL 의 Expected Graph 구성:
    현재 모델 임베딩으로 전체 (Compound, Disease) 쌍을 채점하여
    confidence 가 높은 top-K 쌍을 pseudo-positive 로 추가한다.

    Args:
        z_dict: 이전 outer epoch 에서 계산된 노드 임베딩
        orig_edge_index: 원본 train treats 엣지 (마스킹 대상)
        epoch: 현재 outer epoch (≥2, 에폭에 따라 선형적으로 더 많이 확장)
        growth_rate: epoch 당 확장 비율 r (r·|E_orig|·(epoch−1))
        max_edge_ratio: 원본 대비 최대 확장 비율 cap
        confidence_threshold: sigmoid(score) 가 이 값 이상인 쌍만 인정

    Returns:
        expanded_edge: [2, K] 확장 엣지
        expanded_weight: [K] soft confidence (0~1)
    """
    device = orig_edge_index.device
    raw_scores = model.score_all(z_dict)
    probs = torch.sigmoid(raw_scores)

    # 이미 알려진 positive 는 top-K 후보에서 제외
    probs[orig_edge_index[0], orig_edge_index[1]] = 0.0

    n_orig = orig_edge_index.shape[1]
    target_n = int(n_orig * growth_rate * (epoch - 1))
    max_allowed = int(n_orig * max_edge_ratio)
    n_add = min(target_n, max_allowed)

    if n_add <= 0:
        empty_e = torch.empty((2, 0), dtype=torch.long, device=device)
        empty_w = torch.empty((0,), device=device)
        return empty_e, empty_w

    flat_probs = probs.flatten()

    # Confidence threshold 로 품질 하한 보장
    above_thr = (flat_probs > confidence_threshold).sum().item()
    if above_thr < n_add:
        n_add = int(above_thr)

    if n_add <= 0:
        empty_e = torch.empty((2, 0), dtype=torch.long, device=device)
        empty_w = torch.empty((0,), device=device)
        return empty_e, empty_w

    topk_vals, topk_idx = torch.topk(flat_probs, n_add)
    n_disease = probs.shape[1]
    rows = torch.div(topk_idx, n_disease, rounding_mode='floor')
    cols = topk_idx % n_disease

    expanded_edge = torch.stack([rows, cols], dim=0)
    expanded_weight = topk_vals  # soft confidence
    return expanded_edge, expanded_weight


def train(model, optimizer, data, train_data, epoch, z_dict_prev=None,
          inner_steps=50, growth_rate=0.03, max_edge_ratio=1.0,
          confidence_threshold=0.85, lambda_c=1.0):
    """
    HeteroPULL 의 한 outer epoch.

    PULL 의 dual loss:
        L_E : Expected Graph (원본 + 확장 pseudo-positive) 위의 BCE.
              expanded 엣지는 soft confidence 로 가중.
        L_C : 원본 treats 엣지만으로 계산하는 correction BCE.
              확장이 잘못된 방향으로 drift 하지 않도록 ground truth 로 anchoring.
        L   = L_E + λ_c · L_C

    Epoch 1 에서는 사전 임베딩이 없어 확장 없이 원본만 사용 (L_E ≈ L_C).
    """
    device = train_data[EDGE_TYPE].edge_index.device
    orig_edge = train_data[EDGE_TYPE].edge_index  # [2, N_orig]
    n_c = data['Compound'].num_nodes
    n_d = data['Disease'].num_nodes

    # 1) Expected Graph 구성 (epoch ≥ 2 부터)
    if epoch > 1 and z_dict_prev is not None:
        expanded_edge, expanded_weight = expand_graph(
            model, z_dict_prev, orig_edge, epoch,
            growth_rate=growth_rate,
            max_edge_ratio=max_edge_ratio,
            confidence_threshold=confidence_threshold,
        )
    else:
        expanded_edge = torch.empty((2, 0), dtype=torch.long, device=device)
        expanded_weight = torch.empty((0,), device=device)

    n_exp = expanded_edge.size(1)
    expected_edge = torch.cat([orig_edge, expanded_edge], dim=1)  # L_E positives

    # 2) Encoder 입력: expected graph (treats = 원본 + 확장)
    edge_index_dict = _build_edge_index_dict(train_data, expected_edge)

    last_loss = 0.0
    z_dict_new = None
    for _ in range(inner_steps):
        model.train()
        optimizer.zero_grad()

        z_dict_new = model.encode(data, edge_index_dict)

        # -------- L_E : Expected Graph Loss --------
        neg_e = negative_sampling(
            edge_index=expected_edge,
            num_nodes=(n_c, n_d),
            num_neg_samples=expected_edge.size(1),
            method='sparse',
        ).to(device)

        pos_logits_e = model.decode(z_dict_new, expected_edge)
        neg_logits_e = model.decode(z_dict_new, neg_e)

        logits_e = torch.cat([pos_logits_e, neg_logits_e])
        labels_e = torch.cat([
            torch.ones(pos_logits_e.size(0), device=device),
            torch.zeros(neg_logits_e.size(0), device=device),
        ])
        # 원본 엣지는 weight 1.0, 확장 엣지는 soft weight, negative 는 1.0
        weights_e = torch.cat([
            torch.ones(orig_edge.size(1), device=device),
            expanded_weight,
            torch.ones(neg_logits_e.size(0), device=device),
        ])
        loss_e = F.binary_cross_entropy_with_logits(
            logits_e, labels_e, weight=weights_e
        )

        # -------- L_C : Correction Loss (원본만) --------
        neg_c = negative_sampling(
            edge_index=orig_edge,
            num_nodes=(n_c, n_d),
            num_neg_samples=orig_edge.size(1),
            method='sparse',
        ).to(device)

        pos_logits_c = model.decode(z_dict_new, orig_edge)
        neg_logits_c = model.decode(z_dict_new, neg_c)
        logits_c = torch.cat([pos_logits_c, neg_logits_c])
        labels_c = torch.cat([
            torch.ones(pos_logits_c.size(0), device=device),
            torch.zeros(neg_logits_c.size(0), device=device),
        ])
        loss_c = F.binary_cross_entropy_with_logits(logits_c, labels_c)

        loss = loss_e + lambda_c * loss_c
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        last_loss = float(loss.detach().cpu())

    # 다음 epoch 의 graph expansion 에 사용할 임베딩 반환
    return last_loss, z_dict_new, n_exp


@torch.no_grad()
def test(data, model, split_edges, mp_edge_index_dict):
    """
    split_edges: {'index': [2, E], 'label': [E] (1 positive, 0 negative)}
    mp_edge_index_dict: encoder 입력. val/test positive 가 포함되지 않아야 한다.
    """
    model.eval()
    z_dict = model.encode(data, mp_edge_index_dict, None)

    logits = model.decode(z_dict, split_edges['index'])
    labels = split_edges['label'].float().to(logits.device)

    loss = F.binary_cross_entropy_with_logits(logits, labels)
    auc = roc_auc_score(labels.cpu().numpy(),
                        torch.sigmoid(logits).cpu().numpy())
    return loss, auc


@torch.no_grad()
def get_drug_repurposing_candidates(data, model, mp_edge_index_dict,
                                    num_candidates=20,
                                    max_per_disease=3, min_prob=0.5):
    print("\n[약물 재창출 후보 분석 시작]")
    model.eval()
    z_dict = model.encode(data, mp_edge_index_dict, None)

    raw_scores = model.score_all(z_dict).cpu()

    for split in ['train', 'val', 'test']:
        key_name = f'{split}_pos_edge_index'
        if key_name in data[EDGE_TYPE]:
            edge_index = data[EDGE_TYPE][key_name].cpu()
            raw_scores[edge_index[0], edge_index[1]] = -float('inf')

    probs = torch.sigmoid(raw_scores)
    initial_candidates = num_candidates * 10
    flat_probs = probs.flatten()
    top_k_probs, top_k_indices = torch.topk(flat_probs, initial_candidates)

    n_disease = probs.shape[1]
    row_indices = torch.div(top_k_indices, n_disease, rounding_mode='floor')
    col_indices = top_k_indices % n_disease

    compound_names = data.node_names['Compound']
    disease_names = data.node_names['Disease']

    disease_count = {}
    final_candidates = []

    print("\n--- 새로운 약물 재창출 후보 Top 20 ---")
    for i in range(initial_candidates):
        prob = top_k_probs[i].item()
        if prob < min_prob:
            continue

        c_name = compound_names[row_indices[i].item()]
        d_name = disease_names[col_indices[i].item()]

        if disease_count.get(d_name, 0) >= max_per_disease:
            continue
        disease_count[d_name] = disease_count.get(d_name, 0) + 1

        final_candidates.append({'compound': c_name, 'disease': d_name, 'prob': prob})
        if len(final_candidates) >= num_candidates:
            break

    for i, item in enumerate(final_candidates):
        print(f"{i+1:02d}. [약물] {item['compound']} -> [질병] {item['disease']} (확률: {item['prob']:.4f})")

    return final_candidates
