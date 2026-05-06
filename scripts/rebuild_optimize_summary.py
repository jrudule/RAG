"""
Atjauno optimize_loop_summary.csv no esošajām opt_<q>_baseline un opt_<q>_candidate mapēm.

Lietojums: python -m scripts.rebuild_optimize_summary
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / 'reports'
SUMMARY_PATH = REPORTS_DIR / 'optimize_loop_summary.csv'


def dir_to_target_q(dir_name, suffix):
    inner = dir_name[len('opt_'):-len(suffix)]
    return inner.replace('_', '.')


def compute_accuracy(csv_path, q_nr):
    if not csv_path.exists():
        return None, 0
    df = pd.read_csv(csv_path, dtype={'Nr': str})
    df['Atbilde'] = df['Atbilde'].astype(str).str.strip().str.lower()
    df['Sagaidama_atbilde'] = df['Sagaidāmā atbilde'].astype(str).str.strip().str.lower()
    df = df[(df['Nr'].astype(str) == str(q_nr)) & (df['Sagaidama_atbilde'] != '?')]
    if len(df) == 0:
        return None, 0
    correct = (df['Atbilde'] == df['Sagaidama_atbilde']).sum()
    return round(correct / len(df) * 100, 2), len(df)


def derive_status(acc_b, n_b, acc_c, candidate_exists):
    if not candidate_exists:
        if acc_b is None:
            return 'no_data'
        if acc_b == 100.0:
            return 'no_failures'
        return 'no_candidate'
    if acc_b is None or acc_c is None:
        return 'no_data'
    if acc_c > acc_b:
        return 'better'
    return 'not_better'


def main():
    baseline_dirs = sorted(p for p in REPORTS_DIR.iterdir()
                           if p.is_dir() and p.name.startswith('opt_') and p.name.endswith('_baseline'))

    rows = []
    for b_dir in baseline_dirs:
        target_q = dir_to_target_q(b_dir.name, '_baseline')
        c_dir = REPORTS_DIR / f'opt_{target_q.replace(".", "_")}_candidate'

        b_csv = b_dir / 'report.csv'
        c_csv = c_dir / 'report.csv'

        acc_b, n_b = compute_accuracy(b_csv, target_q)
        acc_c, n_c = (None, 0)
        if c_csv.exists():
            acc_c, n_c = compute_accuracy(c_csv, target_q)

        status = derive_status(acc_b, n_b, acc_c, c_csv.exists())
        delta = round(acc_c - acc_b, 2) if (acc_b is not None and acc_c is not None) else None

        rows.append({
            'target_q': target_q,
            'baseline_acc': acc_b,
            'baseline_n': n_b,
            'candidate_acc': acc_c,
            'candidate_n': n_c,
            'delta': delta,
            'status': status,
        })

    df = pd.DataFrame(rows)
    df['_sort_a'] = df['target_q'].str.split('.').str[0].astype(float)
    df['_sort_b'] = df['target_q'].apply(
        lambda s: float(s.split('.', 1)[1].replace('-', '.')) if '.' in s else -1.0
    )
    df = df.sort_values(['_sort_a', '_sort_b']).drop(columns=['_sort_a', '_sort_b']).reset_index(drop=True)

    df.to_csv(SUMMARY_PATH, index=False, encoding='utf-8')
    print(df.to_string(index=False))
    print(f'\nSaglabāts: {SUMMARY_PATH}')


if __name__ == '__main__':
    main()
