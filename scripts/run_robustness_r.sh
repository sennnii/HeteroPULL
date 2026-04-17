#!/usr/bin/env bash
# Robustness experiment: 큰 r 값에서의 성능 변화
# τ=0.90 고정, r = {0.05, 0.10, 0.15}
# With L_C (lambda_c=1.0) vs Without L_C (lambda_c=0)
#
# 기존 결과 (r=0.01~0.03)는 sweep_expansion에 있음
# 여기서는 r를 더 키워서 robustness 확인
#
# 사용법:
#   bash scripts/run_robustness_r.sh [GPU_ID]
#
# 결과: results/robustness_r/{with_lc,no_lc}/summary.json

set -e

GPU=${1:-1}
COMMON="--epochs 50 --patience 15 --hidden_dim 128 --out_dim 64 --heads 4 --layers 2"
SEEDS="0 1 2"
TAU=0.90

echo "=========================================="
echo "Robustness Experiment: large r values"
echo "GPU: $GPU, τ=$TAU"
echo "=========================================="

mkdir -p logs/robustness_r
mkdir -p results/robustness_r

# ── With L_C (lambda_c=1.0) ──
echo ""
echo "############################################"
echo "# Phase 1: With L_C (lambda_c=1.0)"
echo "############################################"
CUDA_VISIBLE_DEVICES=$GPU python scripts/run_expansion_sweep.py \
    --seeds $SEEDS \
    --taus $TAU \
    --rs 0.05 0.10 0.15 \
    --out_root results/robustness_r/with_lc \
    --skip_existing \
    -- $COMMON --lambda_c 1.0

# ── Without L_C (lambda_c=0) ──
echo ""
echo "############################################"
echo "# Phase 2: Without L_C (lambda_c=0)"
echo "############################################"
CUDA_VISIBLE_DEVICES=$GPU python scripts/run_expansion_sweep.py \
    --seeds $SEEDS \
    --taus $TAU \
    --rs 0.05 0.10 0.15 \
    --out_root results/robustness_r/no_lc \
    --skip_existing \
    -- $COMMON --lambda_c 0.0

# ── 결과 요약 ──
echo ""
echo "=========================================="
echo "최종 결과 요약"
echo "=========================================="

python3 -c "
import json, os

def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

# 기존 r=0.01~0.03 (with L_C) from sweep
print()
print('=== With L_C (lambda_c=1.0), τ=0.90 ===')
print(f'{\"r\":>6}  {\"test_auc\":>16}  {\"test_auprc\":>16}  {\"test_mrr\":>16}  {\"test_h10\":>16}')
print('-' * 76)

# 기존 sweep 결과 (r=0.01~0.03)
sweep = load('results/sweep_expansion/sweep_summary.json')
if sweep:
    for r in [0.01, 0.02, 0.03]:
        key = f'tau=0.9,r={r}'
        cell = sweep.get('cells', {}).get(key)
        if cell:
            s = cell['summary']
            print(f'{r:>6.2f}  '
                  f'{s[\"test_auc\"][\"mean\"]:>7.4f}±{s[\"test_auc\"][\"std\"]:<7.4f}  '
                  f'{s[\"test_auprc\"][\"mean\"]:>7.4f}±{s[\"test_auprc\"][\"std\"]:<7.4f}  '
                  f'{s[\"test_mrr\"][\"mean\"]:>7.4f}±{s[\"test_mrr\"][\"std\"]:<7.4f}  '
                  f'{s[\"test_h10\"][\"mean\"]:>7.4f}±{s[\"test_h10\"][\"std\"]:<7.4f}')

# 새 결과 (r=0.05~0.15)
for r in [0.05, 0.10, 0.15]:
    tag = f'tau0p90_r{r:.3f}'.replace('.', 'p')
    p = f'results/robustness_r/with_lc/{tag}/summary.json'
    d = load(p)
    if d:
        s = d['summary']
        print(f'{r:>6.2f}  '
              f'{s[\"test_auc\"][\"mean\"]:>7.4f}±{s[\"test_auc\"][\"std\"]:<7.4f}  '
              f'{s[\"test_auprc\"][\"mean\"]:>7.4f}±{s[\"test_auprc\"][\"std\"]:<7.4f}  '
              f'{s[\"test_mrr\"][\"mean\"]:>7.4f}±{s[\"test_mrr\"][\"std\"]:<7.4f}  '
              f'{s[\"test_h10\"][\"mean\"]:>7.4f}±{s[\"test_h10\"][\"std\"]:<7.4f}')
    else:
        print(f'{r:>6.2f}  (not found: {p})')

print()
print('=== Without L_C (lambda_c=0), τ=0.90 ===')
print(f'{\"r\":>6}  {\"test_auc\":>16}  {\"test_auprc\":>16}  {\"test_mrr\":>16}  {\"test_h10\":>16}')
print('-' * 76)

# 기존 no_lc 결과 (r=0.03)
no_lc_base = load('results/full_exp/no_lc_t90/summary.json')
if no_lc_base:
    s = no_lc_base['summary']
    print(f'{0.03:>6.2f}  '
          f'{s[\"test_auc\"][\"mean\"]:>7.4f}±{s[\"test_auc\"][\"std\"]:<7.4f}  '
          f'{s[\"test_auprc\"][\"mean\"]:>7.4f}±{s[\"test_auprc\"][\"std\"]:<7.4f}  '
          f'{s[\"test_mrr\"][\"mean\"]:>7.4f}±{s[\"test_mrr\"][\"std\"]:<7.4f}  '
          f'{s[\"test_h10\"][\"mean\"]:>7.4f}±{s[\"test_h10\"][\"std\"]:<7.4f}')

for r in [0.05, 0.10, 0.15]:
    tag = f'tau0p90_r{r:.3f}'.replace('.', 'p')
    p = f'results/robustness_r/no_lc/{tag}/summary.json'
    d = load(p)
    if d:
        s = d['summary']
        print(f'{r:>6.2f}  '
              f'{s[\"test_auc\"][\"mean\"]:>7.4f}±{s[\"test_auc\"][\"std\"]:<7.4f}  '
              f'{s[\"test_auprc\"][\"mean\"]:>7.4f}±{s[\"test_auprc\"][\"std\"]:<7.4f}  '
              f'{s[\"test_mrr\"][\"mean\"]:>7.4f}±{s[\"test_mrr\"][\"std\"]:<7.4f}  '
              f'{s[\"test_h10\"][\"mean\"]:>7.4f}±{s[\"test_h10\"][\"std\"]:<7.4f}')
    else:
        print(f'{r:>6.2f}  (not found: {p})')
"

echo ""
echo "완료!"
