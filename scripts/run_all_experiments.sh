#!/usr/bin/env bash
# KIISE 학부생 논문용 full experiment sweep.
# 5 설정 × 5 seeds = 25 runs.
#
# 사용법:
#   bash scripts/run_all_experiments.sh
#
# 각 설정별 결과:
#   results/full/summary.json            — HeteroPULL (Ours)
#   results/baseline_hgt/summary.json    — HGT-only baseline
#   results/ablation_no_lc/summary.json  — w/o L_C
#   results/ablation_no_exp/summary.json — w/o graph expansion
#   results/ablation_no_morgan/summary.json — w/o Morgan FP

set -e

COMMON="--epochs 50 --patience 15 --hidden_dim 128 --out_dim 64 --heads 4 --layers 2"
SEEDS="0 1 2 3 4"

# (1) HeteroPULL full (Ours)
python scripts/run_seeds.py \
    --seeds $SEEDS \
    --out_dir results/full \
    --log_dir logs/full \
    -- $COMMON

# (2) HGT-only baseline (no expansion, no L_C)
python scripts/run_seeds.py \
    --seeds $SEEDS \
    --out_dir results/baseline_hgt \
    --log_dir logs/baseline_hgt \
    -- $COMMON --growth_rate 0 --lambda_c 0

# (3) Ablation: w/o L_C
python scripts/run_seeds.py \
    --seeds $SEEDS \
    --out_dir results/ablation_no_lc \
    --log_dir logs/ablation_no_lc \
    -- $COMMON --lambda_c 0

# (4) Ablation: w/o graph expansion
python scripts/run_seeds.py \
    --seeds $SEEDS \
    --out_dir results/ablation_no_exp \
    --log_dir logs/ablation_no_exp \
    -- $COMMON --growth_rate 0

# (5) Ablation: w/o Morgan fingerprint
python scripts/run_seeds.py \
    --seeds $SEEDS \
    --out_dir results/ablation_no_morgan \
    --log_dir logs/ablation_no_morgan \
    -- $COMMON --no_morgan

echo ""
echo "=========================================="
echo "All experiments complete."
echo "=========================================="
for d in full baseline_hgt ablation_no_lc ablation_no_exp ablation_no_morgan; do
    echo ""
    echo "--- $d ---"
    python -c "
import json
with open('results/$d/summary.json') as f:
    s = json.load(f)['summary']
for k in ['test_auc','test_auprc','test_mrr','test_h1','test_h3','test_h10']:
    print(f'  {k:<12} {s[k][\"mean\"]:.4f} ± {s[k][\"std\"]:.4f}')
"
done
