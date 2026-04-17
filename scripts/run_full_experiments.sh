#!/usr/bin/env bash
# HeteroPULL 전체 실험 재실행 (5 seeds)
# GPU 0, 1 병렬 사용
#
# 실험 목록 (8종):
#   HeteroPULL 계열 (main_hetionet.py):
#     1. Full (t=0.90)
#     2. HGT Baseline
#     3. w/o L_C
#     4. w/o Expansion
#     5. w/o Morgan
#   외부 Baseline (main_gcn_baseline.py):
#     6. GCN
#     7. GAT (hidden=64, 메모리 제약)
#     8. SAGE
#
# 사용법:
#   bash scripts/run_full_experiments.sh
#
# 결과: results/full_exp/{exp_name}/summary.json

set -e

COMMON="--epochs 50 --patience 15 --hidden_dim 128 --out_dim 64 --heads 4 --layers 2"
SEEDS="0 1 2 3 4"

mkdir -p logs/full_exp
mkdir -p results/full_exp

# ─────────────────────────────────────────
# HeteroPULL 계열 (run_seeds.py 사용)
# ─────────────────────────────────────────

run_hetero() {
    local name=$1
    local gpu=$2
    shift 2
    local extra="$@"
    echo ""
    echo "========================================"
    echo "[$(date +%H:%M:%S)] Running: $name (GPU $gpu)"
    echo "========================================"
    CUDA_VISIBLE_DEVICES=$gpu python scripts/run_seeds.py \
        --seeds $SEEDS \
        --out_dir results/full_exp/$name \
        --log_dir logs/full_exp/$name \
        -- $COMMON $extra
}

# ─────────────────────────────────────────
# GCN 계열 baseline (개별 실행)
# ─────────────────────────────────────────

run_baseline() {
    local model=$1
    local gpu=$2
    local extra=$3
    echo ""
    echo "========================================"
    echo "[$(date +%H:%M:%S)] Running: ${model}_baseline (GPU $gpu)"
    echo "========================================"
    mkdir -p results/full_exp/${model}_baseline
    for seed in $SEEDS; do
        echo "[${model}] seed=$seed"
        python main_gcn_baseline.py \
            --model $model --seed $seed --gpu $gpu \
            --epochs 100 --patience 15 $extra \
            --result_json results/full_exp/${model}_baseline/seed${seed}.json \
            > logs/full_exp/${model}_baseline_seed${seed}.log 2>&1
    done
    # Aggregate
    python -c "
import json, os
import numpy as np
files = sorted(os.listdir('results/full_exp/${model}_baseline'))
files = [f for f in files if f.startswith('seed')]
if not files:
    print('no results')
else:
    all_metrics = {}
    for f in files:
        with open(f'results/full_exp/${model}_baseline/{f}') as fp:
            d = json.load(fp)
        for k, v in d['final'].items():
            all_metrics.setdefault(k, []).append(v)
    summary = {'n_runs': len(files), 'summary': {
        k: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'values': v}
        for k, v in all_metrics.items()
    }}
    with open('results/full_exp/${model}_baseline/summary.json', 'w') as fp:
        json.dump(summary, fp, indent=2)
    print(f'[{len(files)} seeds]')
    for k in ['test_auc','test_auprc','test_mrr','test_h10']:
        if k in all_metrics:
            print(f'  {k}: {np.mean(all_metrics[k]):.4f} ± {np.std(all_metrics[k]):.4f}')
"
}

# ─────────────────────────────────────────
# GPU 0: HeteroPULL 계열 (Full, w/o L_C, w/o Expansion, GCN)
# GPU 1: 나머지 (HGT Baseline, w/o Morgan, GAT, SAGE)
# 두 개의 백그라운드 그룹으로 실행
# ─────────────────────────────────────────

(
    echo "=== GPU 0 그룹 시작 ==="
    run_hetero "full_t90"     0 --confidence_threshold 0.90
    run_hetero "no_lc_t90"    0 --confidence_threshold 0.90 --lambda_c 0
    run_hetero "no_exp"       0 --growth_rate 0
    run_baseline "gcn"        0 ""
    echo "=== GPU 0 그룹 완료 ==="
) > logs/full_exp/gpu0_group.log 2>&1 &
GPU0_PID=$!
echo "GPU 0 그룹 시작 (PID: $GPU0_PID)"

(
    echo "=== GPU 1 그룹 시작 ==="
    run_hetero "hgt_baseline" 1 --growth_rate 0 --lambda_c 0
    run_hetero "no_morgan_t90" 1 --confidence_threshold 0.90 --no_morgan
    run_baseline "sage"       1 ""
    run_baseline "gat"        1 "--hidden_dim 64 --out_dim 32"
    echo "=== GPU 1 그룹 완료 ==="
) > logs/full_exp/gpu1_group.log 2>&1 &
GPU1_PID=$!
echo "GPU 1 그룹 시작 (PID: $GPU1_PID)"

echo ""
echo "두 그룹 병렬 실행 중..."
echo "  GPU 0 로그: tail -f logs/full_exp/gpu0_group.log"
echo "  GPU 1 로그: tail -f logs/full_exp/gpu1_group.log"
echo ""

# 두 그룹 대기
wait $GPU0_PID
echo "✅ GPU 0 그룹 완료"
wait $GPU1_PID
echo "✅ GPU 1 그룹 완료"

# ─────────────────────────────────────────
# 최종 집계 출력
# ─────────────────────────────────────────
echo ""
echo "=========================================="
echo "모든 실험 완료 — 최종 결과"
echo "=========================================="
for d in full_t90 hgt_baseline no_lc_t90 no_exp no_morgan_t90 gcn_baseline gat_baseline sage_baseline; do
    if [ -f "results/full_exp/$d/summary.json" ]; then
        echo ""
        echo "--- $d ---"
        python -c "
import json
with open('results/full_exp/$d/summary.json') as f:
    s = json.load(f)['summary']
for k in ['test_auc','test_auprc','test_mrr','test_h1','test_h3','test_h10']:
    if k in s:
        print(f'  {k:<12} {s[k][\"mean\"]:.4f} ± {s[k][\"std\"]:.4f}')
"
    fi
done
