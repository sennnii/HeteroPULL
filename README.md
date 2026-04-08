# HeteroPULL: PU Learning for Drug Repurposing on Heterogeneous Biomedical Graphs

PULL (Kim et al., AAAI 2025) 을 이종 그래프 환경(Hetionet)으로 확장하여 약물 재창출(drug repurposing) 후보 약물–질병 쌍을 발굴하는 링크 예측 모델.

## 핵심 기여 (논문 반영 포인트)

### 1. HGT 기반 이종 메시지 패싱
- **HGTConv** (Hu et al., WWW 2020) 를 encoder 로 채택. 노드/엣지 타입별 `W_msg, W_att, W_v` 를 학습하고 type-specific attention softmax 로 메시지를 정규화하므로, 단순 `HeteroConv(sum)` 에서 발생하는 **root feature over-counting** 문제가 구조적으로 발생하지 않는다.
- `src/model_hetionet.py` 의 `HeteroPULLModel.encode()` 참조.

### 2. Target Leakage 방지 — Disjoint MP / Supervision Split
- 링크 예측에서 자주 발생하는 leakage: supervision edge 가 encoder 의 message passing 그래프에도 포함되는 경우 (Hu et al., OGB).
- `main_hetionet.py` 의 `RandomLinkSplit(disjoint_train_ratio=0.3)` 으로 학습용 엣지를 (a) MP 전용 70%, (b) supervision 전용 30% 로 분리.
- `src/train_hetionet.py` 의 `_build_mp_edge_dict()` 가 encoder 입력에서 supervision edge 를 명시적으로 제거하여 이중 보강.

### 3. Non-negative PU Loss
- 약물–질병 그래프에서 unlabeled 쌍은 음성이 아니라 **missing positive** 일 가능성이 높다. 단순 BCE + random negative sampling 은 biased risk estimator 이다.
- **nnPU loss** (Kiryo, Niu, du Plessis, Sugiyama; NeurIPS 2017) 도입:

  $$R_{\text{pu}}(f) = \pi_p \cdot R_p^+(f) + \max\{0,\; R_u^-(f) - \pi_p \cdot R_p^-(f)\}$$

  - $R_p^+ = \mathbb{E}_{x \sim P}[\ell(f(x), +1)]$
  - $R_p^- = \mathbb{E}_{x \sim P}[\ell(f(x), -1)]$
  - $R_u^- = \mathbb{E}_{x \sim U}[\ell(f(x), -1)]$
  - $\ell$: logistic surrogate (softplus)
- Non-negative correction (Eq.6 of Kiryo et al.): negative risk 가 $-\beta$ 미만이 되면 $-\gamma \cdot (R_u^- - \pi_p R_p^-)$ 만 backprop 하여 over-fitting 시 risk 발산을 막는다.
- class prior $\pi_p \approx |\text{treats}| / (|\text{Compound}| \times |\text{Disease}|)$ 로 추정.
- `src/train_hetionet.py` 의 `nnpu_loss()`, `estimate_class_prior()` 참조.

### 4. Heuristic Pseudo-labeling 제거
- 기존 PULL 의 top-K graph expansion 은 self-training 에 가까운 휴리스틱이었음. HeteroPULL 에서는 이를 제거하고 nnPU 의 unbiased risk estimator 로 대체하여 수학적으로 방어 가능한 PU learning 로 전환.

## 코드 구조
```
.
├── main_hetionet.py          # 학습/평가 엔트리 포인트
├── preprocess_hetionet.py    # Hetionet JSON → PyG HeteroData 전처리
├── src/
│   ├── model_hetionet.py     # HeteroPULLModel (HGT + inner-product decoder)
│   └── train_hetionet.py     # nnPU training loop, test, candidate ranking
└── data/
    └── hetionet_data.pt      # 전처리 결과 (gitignored)
```

## 환경
- `python >= 3.8`
- `pytorch >= 1.13`
- `torch-geometric >= 2.3`
- `scikit-learn`

## 실행
```bash
python preprocess_hetionet.py
python main_hetionet.py --epochs 10 --lr 0.01 --hidden_dim 128 --out_dim 64 --heads 4 --layers 2
```

## 주요 하이퍼파라미터
| 이름 | 기본값 | 설명 |
|---|---|---|
| `--hidden_dim` | 128 | HGT hidden size |
| `--out_dim` | 64 | 최종 임베딩 차원 |
| `--heads` | 4 | HGT attention heads |
| `--layers` | 2 | HGT layer 수 |
| `--lr` | 0.01 | Adam learning rate |
| `inner_steps` | 50 | epoch 당 inner optimization step |
| `unl_ratio` | 5 | positive 대비 unlabeled batch 배수 |

## 참고문헌
- Kim, J., Park, K. H., Yoon, H., & Kang, U. (2025). **Accurate Link Prediction for Edge-Incomplete Graphs via PU Learning.** *AAAI*.
- Hu, Z., Dong, Y., Wang, K., & Sun, Y. (2020). **Heterogeneous Graph Transformer.** *WWW*.
- Kiryo, R., Niu, G., du Plessis, M. C., & Sugiyama, M. (2017). **Positive-Unlabeled Learning with Non-Negative Risk Estimator.** *NeurIPS*.
- Himmelstein, D. S., et al. (2017). **Systematic integration of biomedical knowledge prioritizes drugs for repurposing (Hetionet).** *eLife*.
