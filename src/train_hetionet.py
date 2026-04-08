from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling


EDGE_TYPE = ('Compound', 'treats', 'Disease')
REV_EDGE_TYPE = ('Disease', 'rev_treats', 'Compound')


def _surrogate(logits, positive: bool):
    return F.softplus(-logits) if positive else F.softplus(logits)


def nnpu_loss(pos_logits, unl_logits, prior, beta=0.0, gamma=1.0):
    pos_loss_pos = _surrogate(pos_logits, positive=True).mean()
    pos_loss_neg = _surrogate(pos_logits, positive=False).mean()
    unl_loss_neg = _surrogate(unl_logits, positive=False).mean()

    pos_risk = prior * pos_loss_pos
    neg_risk = unl_loss_neg - prior * pos_loss_neg

    if neg_risk.item() < -beta:
        return -gamma * neg_risk, (pos_risk + neg_risk).detach()
    return pos_risk + neg_risk, (pos_risk + neg_risk).detach()


def _build_mp_edge_dict(train_data):
    mp_dict = {k: v for k, v in train_data.edge_index_dict.items()}
    mp_edge = train_data[EDGE_TYPE].edge_index
    mp_dict[EDGE_TYPE] = mp_edge
    if REV_EDGE_TYPE in mp_dict:
        mp_dict[REV_EDGE_TYPE] = mp_edge.flip(0)
    return mp_dict


def estimate_class_prior(data):
    n_pos = data[EDGE_TYPE].edge_index.size(1)
    n_c = data['Compound'].num_nodes
    n_d = data['Disease'].num_nodes
    prior = n_pos / float(n_c * n_d)
    return float(min(max(prior, 1e-4), 0.5))


def train(model, optimizer, data, train_data, criterion, epoch,
          z_dict=None, inner_steps=50, unl_ratio=5):
    device = train_data[EDGE_TYPE].edge_index.device

    sup_edge_idx = train_data[EDGE_TYPE].edge_label_index
    sup_edge_label = train_data[EDGE_TYPE].edge_label
    sup_edge_idx = sup_edge_idx[:, sup_edge_label > 0.5]

    edge_index_dict = _build_mp_edge_dict(train_data)
    prior = estimate_class_prior(data)

    last_loss = 0.0
    z_dict_new = None
    for _ in range(inner_steps):
        model.train()
        optimizer.zero_grad()

        n_unl = sup_edge_idx.size(1) * unl_ratio
        unl_edge_idx = negative_sampling(
            edge_index=train_data[EDGE_TYPE].edge_index,
            num_nodes=(data['Compound'].num_nodes, data['Disease'].num_nodes),
            num_neg_samples=n_unl,
            method='sparse',
        ).to(device)

        z_dict_new = model.encode(data, edge_index_dict)
        pos_logits = model.decode(z_dict_new, sup_edge_idx)
        unl_logits = model.decode(z_dict_new, unl_edge_idx)

        loss, monitored = nnpu_loss(pos_logits, unl_logits, prior=prior)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        last_loss = float(monitored.detach().cpu())

    return last_loss, z_dict_new, sup_edge_idx, None


@torch.no_grad()
def test(data, model, split_edge_index, criterion):
    model.eval()
    z_dict = model.encode(data, data.edge_index_dict, None)

    pos_out = model.decode(z_dict, split_edge_index['pos'])
    neg_out = model.decode(z_dict, split_edge_index['neg'])

    out = torch.cat([pos_out, neg_out]).view(-1)
    edge_label = torch.cat([
        torch.ones(pos_out.size(0)),
        torch.zeros(neg_out.size(0))
    ], dim=0).to(out.device)

    loss = F.binary_cross_entropy_with_logits(out, edge_label)
    auc = roc_auc_score(edge_label.cpu().numpy(),
                        torch.sigmoid(out).cpu().numpy())
    return loss, auc


@torch.no_grad()
def get_drug_repurposing_candidates(data, model, num_candidates=20,
                                    max_per_disease=3, min_prob=0.5):
    print("\n[약물 재창출 후보 분석 시작]")
    model.eval()
    z_dict = model.encode(data, data.edge_index_dict, None)

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

    row_indices = top_k_indices // probs.shape[1]
    col_indices = top_k_indices % probs.shape[1]

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
