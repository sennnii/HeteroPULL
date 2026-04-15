"""
Expansion (τ, r) 하이퍼파라미터 스윕.

목적: "w/o expansion" ablation 이 full 모델을 상회하는 현상을 해결하기 위해,
confidence_threshold (τ) 와 growth_rate (r) 를 바꿔가며 Expansion 이 positive
하게 작동하는 구간이 존재하는지 검증한다.

각 (τ, r) 조합마다 run_seeds.py 를 호출해 N seeds 집계 결과를 얻고, 마지막에
전체 grid 의 test_mrr / test_auc 히트맵 요약을 출력한다.

사용 예시:
    python scripts/run_expansion_sweep.py \\
        --seeds 0 1 2 \\
        --taus 0.85 0.90 0.95 0.99 \\
        --rs 0.01 0.02 0.03 \\
        --out_root results/sweep_expansion \\
        -- --epochs 50 --patience 15 \\
           --hidden_dim 128 --out_dim 64 --heads 4 --layers 2
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Expansion (tau, r) hyperparameter sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2],
                        help="각 (τ,r) 조합에 대해 돌릴 seeds (default: 0 1 2)")
    parser.add_argument('--taus', type=float, nargs='+',
                        default=[0.85, 0.90, 0.95, 0.99],
                        help="confidence_threshold τ 값들")
    parser.add_argument('--rs', type=float, nargs='+',
                        default=[0.01, 0.02, 0.03],
                        help="growth_rate r 값들")
    parser.add_argument('--out_root', type=str, required=True,
                        help="스윕 결과 루트 디렉토리 (하위에 tau{τ}_r{r}/ 생성)")
    parser.add_argument('--python', type=str, default=sys.executable)
    parser.add_argument('--skip_existing', action='store_true',
                        help="summary.json 이 이미 있는 조합은 건너뜀")
    parser.add_argument('extra', nargs=argparse.REMAINDER,
                        help="'--' 이후는 main_hetionet.py 로 그대로 전달")
    return parser.parse_args()


def tag(tau, r):
    return f"tau{tau:.2f}_r{r:.3f}".replace('.', 'p')


def run_one(args, tau, r, extra):
    out_dir = Path(args.out_root) / tag(tau, r)
    summary_path = out_dir / 'summary.json'
    if args.skip_existing and summary_path.exists():
        print(f"[{tag(tau, r)}] summary.json 존재 → skip")
        return summary_path

    cmd = [
        args.python, '-u', 'scripts/run_seeds.py',
        '--seeds', *[str(s) for s in args.seeds],
        '--out_dir', str(out_dir),
        '--log_dir', str(out_dir / 'logs'),
        '--skip_existing',
        '--',
        '--confidence_threshold', str(tau),
        '--growth_rate', str(r),
    ] + extra

    print(f"\n{'#' * 70}")
    print(f"# SWEEP: τ={tau}, r={r}  → {out_dir}")
    print('#' * 70, flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"[{tag(tau, r)}] FAILED (rc={proc.returncode})", file=sys.stderr)
        return None
    return summary_path


def load_summary(path):
    if not path or not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def print_grid(args, summaries):
    keys = ['test_mrr', 'test_auc', 'test_h10', 'test_auprc']
    for key in keys:
        print(f"\n{'=' * 70}")
        print(f"Grid: {key} (mean ± std over seeds)")
        print('=' * 70)
        header = f"{'τ \\ r':>8}  " + "  ".join(f"{r:>14}" for r in args.rs)
        print(header)
        print('-' * len(header))
        for tau in args.taus:
            cells = []
            for r in args.rs:
                s = summaries.get((tau, r))
                if s is None:
                    cells.append(f"{'--':>14}")
                    continue
                m = s['summary'][key]['mean']
                sd = s['summary'][key]['std']
                cells.append(f"{m:>7.4f}±{sd:<5.3f}")
            print(f"{tau:>8.2f}  " + "  ".join(cells))


def main():
    args = parse_args()
    extra = [a for a in args.extra if a != '--']

    print(f"Seeds per cell: {args.seeds}")
    print(f"τ grid: {args.taus}")
    print(f"r grid: {args.rs}")
    print(f"Total cells: {len(args.taus) * len(args.rs)}  "
          f"(총 runs: {len(args.taus) * len(args.rs) * len(args.seeds)})")
    print(f"Extra args → main_hetionet.py: {extra}")

    summaries = {}
    for tau in args.taus:
        for r in args.rs:
            p = run_one(args, tau, r, extra)
            summaries[(tau, r)] = load_summary(p)

    print_grid(args, summaries)

    # 전체 스윕 요약 저장
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    flat = {
        f"tau={tau},r={r}": s for (tau, r), s in summaries.items()
    }
    with open(out_root / 'sweep_summary.json', 'w') as f:
        json.dump({
            'seeds': args.seeds,
            'taus': args.taus,
            'rs': args.rs,
            'extra_args': extra,
            'cells': flat,
        }, f, indent=2)
    print(f"\n[sweep_summary.json 저장] {out_root / 'sweep_summary.json'}")


if __name__ == '__main__':
    main()
