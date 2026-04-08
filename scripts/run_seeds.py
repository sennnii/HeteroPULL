"""
Seed variance / ablation sweep runner.

여러 seed 로 main_hetionet.py 를 subprocess 로 돌려 각 run 의 final metric 을
JSON 으로 수집한 뒤 mean±std 집계 표를 출력한다.

사용 예시:
    # 기본 5 seed variance (기본 하이퍼파라미터)
    python scripts/run_seeds.py \
        --seeds 0 1 2 3 4 \
        --out_dir results/main \
        --log_dir logs/main \
        -- --epochs 50 --patience 15 \
           --hidden_dim 64 --out_dim 32 --heads 2 --layers 2

    # Ablation (--lambda_c 0) 도 동일 스크립트로
    python scripts/run_seeds.py \
        --seeds 0 1 2 3 4 \
        --out_dir results/ablation_no_lc \
        --log_dir logs/ablation_no_lc \
        -- --epochs 50 --patience 15 --lambda_c 0.0 \
           --hidden_dim 64 --out_dim 32 --heads 2 --layers 2

"--" 이후 인자는 전부 main_hetionet.py 로 그대로 전달된다.
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path


METRIC_KEYS = [
    'val_auc', 'val_auprc', 'val_mrr', 'val_h1', 'val_h3', 'val_h10',
    'test_auc', 'test_auprc', 'test_mrr', 'test_h1', 'test_h3', 'test_h10',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Seed sweep runner for HeteroPULL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                        help="실행할 seed 목록 (default: 0 1 2 3 4)")
    parser.add_argument('--out_dir', type=str, required=True,
                        help="각 seed 별 result_json 을 저장할 디렉토리")
    parser.add_argument('--log_dir', type=str, default=None,
                        help="각 seed 별 stdout 로그 저장 디렉토리 (옵션)")
    parser.add_argument('--python', type=str, default=sys.executable,
                        help="사용할 python 실행 파일 (default: 현재 interpreter)")
    parser.add_argument('--skip_existing', action='store_true',
                        help="이미 result_json 이 있는 seed 는 건너뜀 (resume 용)")
    parser.add_argument('extra', nargs=argparse.REMAINDER,
                        help="'--' 이후는 main_hetionet.py 로 그대로 전달")
    return parser.parse_args()


def mean_std(xs):
    n = len(xs)
    if n == 0:
        return float('nan'), float('nan')
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)  # sample std
    return m, math.sqrt(var)


def run_one_seed(args, seed, extra_args):
    out_path = Path(args.out_dir) / f"seed_{seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and out_path.exists():
        print(f"[seed {seed}] 이미 존재: {out_path} → skip")
        return out_path

    cmd = [
        args.python, '-u', 'main_hetionet.py',
        '--seed', str(seed),
        '--result_json', str(out_path),
        '--skip_candidates',
    ] + extra_args

    print(f"\n{'=' * 70}")
    print(f"[seed {seed}] {' '.join(cmd)}")
    print('=' * 70, flush=True)

    if args.log_dir:
        log_path = Path(args.log_dir) / f"seed_{seed}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'w') as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        print(f"[seed {seed}] log → {log_path}")
    else:
        proc = subprocess.run(cmd)

    if proc.returncode != 0:
        print(f"[seed {seed}] FAILED (returncode={proc.returncode})",
              file=sys.stderr)
        return None
    if not out_path.exists():
        print(f"[seed {seed}] result_json 이 생성되지 않았음: {out_path}",
              file=sys.stderr)
        return None
    return out_path


def load_results(out_dir, seeds):
    results = []
    for s in seeds:
        p = Path(out_dir) / f"seed_{s}.json"
        if not p.exists():
            print(f"[warn] {p} not found — 집계에서 제외", file=sys.stderr)
            continue
        with open(p) as f:
            results.append(json.load(f))
    return results


def print_summary(results):
    if not results:
        print("집계할 결과가 없습니다.")
        return

    seeds = [r['seed'] for r in results]
    best_epochs = [r['best_epoch'] for r in results]

    print(f"\n{'=' * 70}")
    print(f"Seed sweep 집계 ({len(results)} runs, seeds={seeds})")
    print('=' * 70)

    print(f"\nBest epoch 분포: {best_epochs}  "
          f"(mean={sum(best_epochs) / len(best_epochs):.1f})")

    # Per-seed 전체 metric
    print("\n[Per-seed final metrics]")
    header = f"{'seed':>5}  {'best_ep':>7}  " + "  ".join(
        f"{k:>9}" for k in METRIC_KEYS
    )
    print(header)
    print('-' * len(header))
    for r in results:
        row = f"{r['seed']:>5}  {r['best_epoch']:>7}  " + "  ".join(
            f"{r['final'][k]:>9.4f}" for k in METRIC_KEYS
        )
        print(row)

    # Mean ± std 요약
    print("\n[Summary (mean ± std over seeds)]")
    print(f"{'metric':<12}  {'mean':>8}  {'std':>8}")
    print('-' * 34)
    summary = {}
    for k in METRIC_KEYS:
        xs = [r['final'][k] for r in results]
        m, s = mean_std(xs)
        summary[k] = {'mean': m, 'std': s, 'values': xs}
        print(f"{k:<12}  {m:>8.4f}  {s:>8.4f}")

    return summary


def main():
    args = parse_args()

    # REMAINDER 는 '--' 를 포함할 수 있음 — 제거
    extra_args = [a for a in args.extra if a != '--']

    print(f"Python: {args.python}")
    print(f"Seeds: {args.seeds}")
    print(f"Out dir: {args.out_dir}")
    print(f"Log dir: {args.log_dir}")
    print(f"Extra args → main_hetionet.py: {extra_args}")

    for seed in args.seeds:
        run_one_seed(args, seed, extra_args)

    results = load_results(args.out_dir, args.seeds)
    summary = print_summary(results)

    # 집계 결과도 JSON 으로 저장
    if summary is not None:
        summary_path = Path(args.out_dir) / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump({
                'seeds': args.seeds,
                'extra_args': extra_args,
                'n_runs': len(results),
                'summary': summary,
            }, f, indent=2)
        print(f"\n[summary.json 저장] {summary_path}")


if __name__ == '__main__':
    main()
