# HeteroPULL: PU Learning for Drug Repurposing on Heterogeneous Biomedical Graphs

PULL (Kim et al., AAAI 2025) 의 Expected Graph 기반 PU learning 프레임워크를 이종 그래프(Hetionet) 환경으로 확장하여 약물 재창출(drug repurposing) 후보 약물–질병 쌍을 발굴하는 링크 예측 모델.

## 핵심 기여 (논문 반영 포인트)

### 1. PULL 의 Expected Graph 프레임워크를 이종 그래프로 확장
원본 PULL 은 동종(homogeneous) 그래프에서만 정의된다. HeteroPULL 은:
- **HGT (Hu et al., WWW 2020)** 를 encoder 로 채택하여 노드/엣지 타입별 type-specific attention 을 수행.
- PULL 의 두 축 loss — **L_E (Expected Graph Loss)** 와 **L_C (Correction Loss)** — 를 이종 그래프의 treats 관계 (Compound→Disease) 위에서 정의.

$$L = L_E + \lambda_c \cdot L_C$$

- **L_E**: 원본 treats 엣지 + 모델이 confident 하게 예측한 top-K pseudo-positive (확장 엣지) 위에서 BCE. 확장 엣지는 sigmoid 확신도를 soft weight 으로 사용.
- **L_C**: 원본 treats 엣지만으로 계산하는 BCE. 모델이 pseudo-label 에 과적합하지 않도록 ground-truth 에 anchoring.

### 2. Bounded Graph Expansion with Confidence Threshold
매 outer epoch 마다 `|E_exp| = r · |E_orig| · (epoch − 1)` 개의 쌍을 score 상위에서 선택하되:
- `confidence_threshold` 이상인 쌍만 pseudo-positive 로 인정 (품질 하한)
- `max_edge_ratio` 로 누적 확장 총량을 cap (over-expansion 방지)
- 이미 알려진 positive 는 0으로 마스킹 후 top-K

### 3. Target Leakage 방지
- Val/Test 엣지는 RandomLinkSplit 으로 분리되며, encoder 평가 시 `train_data.edge_index_dict` 만 사용해 val/test positive 가 encoder 에 노출되지 않도록 보장.
- Val/Test AUC 계산은 RandomLinkSplit 이 반환한 `edge_label_index` + `edge_label` 을 그대로 사용 (pos/neg 혼합을 올바르게 label 로 구분).

### 4. Compound Morgan Fingerprint Feature
- Hetionet 원본 JSON 의 InChI 로부터 RDKit 으로 **512-bit Morgan fingerprint (radius=2)** 를 생성하여 Compound 노드의 input feature 로 사용.
- Disease 노드는 학습 가능한 embedding 사용.
- `get_compound_features()` 는 `node_mapping['Compound']` 인덱스에 정렬된 행렬을 직접 할당하여 순서 의존성 제거.

## 코드 구조
```
.
├── main_hetionet.py          # 학습/평가 엔트리 포인트
├── preprocess_hetionet.py    # Hetionet JSON → PyG HeteroData 전처리
├── src/
│   ├── model_hetionet.py     # HeteroPULLModel (HGT encoder + inner-product decoder)
│   └── train_hetionet.py     # PULL 학습 루프 (L_E + L_C), graph expansion, test, candidate ranking
└── data/
    ├── hetionet-v1.0.json.bz2 (원본)
    └── hetionet_data.pt       # 전처리 결과 (gitignored)
```

## 환경
- `python >= 3.8`
- `pytorch >= 1.13`
- `torch-geometric >= 2.3`
- `rdkit`
- `scikit-learn`

## 실행
```bash
# 1) 전처리
python preprocess_hetionet.py

# 2) 학습
python main_hetionet.py --epochs 100 --lr 0.003 --hidden_dim 128 --out_dim 64 \
                        --heads 4 --layers 2 --growth_rate 0.03 \
                        --confidence_threshold 0.85 --lambda_c 1.0
```

## 주요 하이퍼파라미터
| 이름 | 기본값 | 설명 |
|---|---|---|
| `--hidden_dim` | 128 | HGT hidden size |
| `--out_dim` | 64 | 최종 임베딩 차원 |
| `--heads` | 4 | HGT attention heads |
| `--layers` | 2 | HGT layer 수 |
| `--lr` | 0.003 | Adam learning rate |
| `--weight_decay` | 1e-4 | L2 regularization |
| `--inner_steps` | 50 | outer epoch 당 inner optimization step |
| `--growth_rate` | 0.03 | epoch 당 확장 비율 r |
| `--max_edge_ratio` | 1.0 | 원본 대비 최대 확장 cap |
| `--confidence_threshold` | 0.85 | pseudo-positive 인정 최소 sigmoid 점수 |
| `--lambda_c` | 1.0 | L_C 가중치 |
| `--patience` | 20 | Early stopping patience |

## 참고문헌
- Kim, J., Park, K. H., Yoon, H., & Kang, U. (2025). **Accurate Link Prediction for Edge-Incomplete Graphs via PU Learning.** *AAAI*.
- Hu, Z., Dong, Y., Wang, K., & Sun, Y. (2020). **Heterogeneous Graph Transformer.** *WWW*.
- Himmelstein, D. S., et al. (2017). **Systematic integration of biomedical knowledge prioritizes drugs for repurposing (Hetionet).** *eLife*.
