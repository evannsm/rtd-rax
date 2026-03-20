"""
Repeated timing benchmark for the Case Study 3 disturbance compare.

This reruns the representative seeded course multiple times, logs the measured
per-planning-cycle timing breakdown for both standard RTD and RTD-RAX, and
emits a manuscript-ready LaTeX table summarizing the mean and sample standard
deviation of the run-level averages.
"""

import argparse
import csv
import json
import math

import numpy as np

from disturbance_case_study_utils import (
    ensure_parent_dir,
    generate_gap_patch_course,
    inset_road_edge_obstacles,
    load_case_study_models,
    run_episode,
)
from immrax_verify import warmup_verifier


TABLE_METRICS = [
    ('solve_setup_time', 'Constraint setup'),
    ('solve_optimize_time', 'RTD solve'),
    ('prepare_time', 'Reference rollout'),
    ('initial_verify_time', 'immrax verify'),
    ('repair_time', 'Repair loop'),
    ('compute_time', 'Total cycle'),
]


def _parse_args():
    p = argparse.ArgumentParser(description='Repeated timing benchmark for Case Study 3')
    p.add_argument('--seed', type=int, default=32)
    p.add_argument('--runs', type=int, default=20)
    p.add_argument('--warmup-runs', type=int, default=12)
    p.add_argument('--course-length', type=float, default=6.0)
    p.add_argument('--road-half-width', type=float, default=1.45)
    p.add_argument('--stages', type=int, default=3)
    p.add_argument('--v0', type=float, default=0.75)
    p.add_argument('--max-steps', type=int, default=55)
    p.add_argument('--goal-tol', type=float, default=0.12)
    p.add_argument('--speed-tol', type=float, default=0.08)
    p.add_argument('--t-move', type=float, default=0.48)
    p.add_argument('--verify-uncertainty', type=float, default=0.01)
    p.add_argument('--verify-dt', type=float, default=0.01)
    p.add_argument('--repair-max-iters', type=int, default=4)
    p.add_argument('--repair-speed-backoff', type=float, default=0.08)
    p.add_argument('--repair-buffer-step', type=float, default=0.015)
    p.add_argument('--obstacle-inset', type=float, default=0.30)
    p.add_argument('--save-csv', type=str, default=None)
    p.add_argument('--save-summary-json', type=str, default=None)
    p.add_argument('--save-table-tex', type=str, default=None)
    return p.parse_args()


def _to_ms(value_s):
    return 1e3 * float(value_s)


def _safe_mean_ms(step_records, key):
    if not step_records:
        return math.nan
    vals = np.array([float(rec[key]) for rec in step_records], dtype=float)
    return float(np.mean(1e3 * vals))


def _sample_std(values):
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1))


def _run_summary(run_index, result):
    summary = {
        'run_index': int(run_index),
        'planner': result['planner'],
        'label': result['label'],
        'status': result['status'],
        'planning_cycles': int(len(result['step_records'])),
        'repair_count': int(result['repair_count']),
        'goal_distance_final_m': float(result['goal_distance_final']),
    }
    for key, _ in TABLE_METRICS:
        summary[f'{key}_mean_ms'] = _safe_mean_ms(result['step_records'], key)
    return summary


def _write_raw_csv(path, rows):
    ensure_parent_dir(path)
    fieldnames = [
        'run_index',
        'planner',
        'label',
        'scenario_seed',
        'episode_status',
        'episode_cycles',
        'episode_repairs',
        'goal_distance_final_m',
        'planning_step',
        'feasible',
        'repair_applied',
        'repair_iters',
        'verify_safe',
        'solve_setup_ms',
        'solve_optimize_ms',
        'solve_total_ms',
        'prepare_ms',
        'initial_verify_ms',
        'verify_total_ms',
        'repair_total_ms',
        'repair_solve_ms',
        'repair_prepare_ms',
        'repair_verify_ms',
        'compute_total_ms',
    ]
    with open(path, 'w', newline='', encoding='ascii') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_run_summaries(run_summaries):
    payload = {}
    for planner in ('standard', 'rtd_rax'):
        planner_runs = [row for row in run_summaries if row['planner'] == planner]
        metric_summary = {}
        for key, label in TABLE_METRICS:
            vals = [row[f'{key}_mean_ms'] for row in planner_runs]
            metric_summary[key] = {
                'label': label,
                'mean_ms': float(np.mean(vals)) if vals else math.nan,
                'std_ms': _sample_std(vals),
            }
        payload[planner] = {
            'runs': len(planner_runs),
            'statuses': {
                status: sum(1 for row in planner_runs if row['status'] == status)
                for status in sorted({row['status'] for row in planner_runs})
            },
            'metrics': metric_summary,
        }
    return payload


def _latex_value(metric_summary):
    mean_ms = metric_summary['mean_ms']
    std_ms = metric_summary['std_ms']
    if not np.isfinite(mean_ms):
        return '--'
    return f'{mean_ms:.2f} $\\\\pm$ {std_ms:.2f}'


def _make_table_tex(args, aggregate):
    warmup_clause = ''
    if int(args.warmup_runs) > 0:
        warmup_clause = (
            f'after {int(args.warmup_runs)} untimed warm-up executions and '
        )
    lines = [
        '\\begin{table}[t]',
        '\\centering',
        (
            f'\\caption{{Case Study 3 timing breakdown {warmup_clause}over {int(args.runs)} measured executions '
            f'of the representative disturbance course (\\texttt{{seed}}={int(args.seed)}). Each entry reports '
            'the mean and sample standard deviation of the per-planning-cycle step time, in '
            f'milliseconds, computed from the {int(args.runs)} run-level averages.}}'
        ),
        '\\label{tab:case3_timing}',
        '\\begin{tabular}{lcc}',
        '\\toprule',
        'Pipeline step & Standard RTD [ms] & RTD-RAX [ms] \\\\',
        '\\midrule',
    ]
    for key, label in TABLE_METRICS:
        std_val = _latex_value(aggregate['standard']['metrics'][key])
        rax_val = _latex_value(aggregate['rtd_rax']['metrics'][key])
        lines.append(f'{label} & {std_val} & {rax_val} \\\\')
    lines.extend([
        '\\bottomrule',
        '\\end{tabular}',
        '\\end{table}',
        '',
    ])
    return '\n'.join(lines)


def main():
    args = _parse_args()
    warmup_verifier(dt=args.verify_dt)

    scenario = generate_gap_patch_course(
        seed=args.seed,
        course_length=args.course_length,
        road_half_width=args.road_half_width,
        stage_count=args.stages,
        stage_width_range=(0.55, 0.75),
        gap_width_range=(0.80, 0.96),
        patch_length_range=(0.80, 1.10),
        patch_gap_range=(0.00, 0.04),
        disturbance_y_range=(0.10, 0.15),
        disturbance_h_range=(0.07, 0.12),
        title='Planning Against Disturbances',
    )
    scenario = inset_road_edge_obstacles(scenario, args.obstacle_inset)
    models = load_case_study_models()

    raw_rows = []
    run_summaries = []
    for _ in range(int(args.warmup_runs)):
        for planner in ('standard', 'rtd_rax'):
            run_episode(
                scenario,
                planner,
                models,
                v0=args.v0,
                max_steps=args.max_steps,
                goal_tol=args.goal_tol,
                speed_tol=args.speed_tol,
                t_move=args.t_move,
                verify_uncertainty=args.verify_uncertainty,
                verify_dt=args.verify_dt,
                repair_max_iters=args.repair_max_iters,
                repair_speed_backoff=args.repair_speed_backoff,
                repair_buffer_step=args.repair_buffer_step,
            )

    for run_index in range(1, int(args.runs) + 1):
        for planner in ('standard', 'rtd_rax'):
            result = run_episode(
                scenario,
                planner,
                models,
                v0=args.v0,
                max_steps=args.max_steps,
                goal_tol=args.goal_tol,
                speed_tol=args.speed_tol,
                t_move=args.t_move,
                verify_uncertainty=args.verify_uncertainty,
                verify_dt=args.verify_dt,
                repair_max_iters=args.repair_max_iters,
                repair_speed_backoff=args.repair_speed_backoff,
                repair_buffer_step=args.repair_buffer_step,
            )
            run_summaries.append(_run_summary(run_index, result))
            for record in result['step_records']:
                raw_rows.append(
                    {
                        'run_index': int(run_index),
                        'planner': result['planner'],
                        'label': result['label'],
                        'scenario_seed': int(args.seed),
                        'episode_status': result['status'],
                        'episode_cycles': int(len(result['step_records'])),
                        'episode_repairs': int(result['repair_count']),
                        'goal_distance_final_m': float(result['goal_distance_final']),
                        'planning_step': int(record['step']),
                        'feasible': bool(record['feasible']),
                        'repair_applied': bool(record['repair_applied']),
                        'repair_iters': int(record['repair_iters']),
                        'verify_safe': '' if record['verify_safe'] is None else bool(record['verify_safe']),
                        'solve_setup_ms': _to_ms(record['solve_setup_time']),
                        'solve_optimize_ms': _to_ms(record['solve_optimize_time']),
                        'solve_total_ms': _to_ms(record['solve_time']),
                        'prepare_ms': _to_ms(record['prepare_time']),
                        'initial_verify_ms': _to_ms(record['initial_verify_time']),
                        'verify_total_ms': _to_ms(record['verify_time']),
                        'repair_total_ms': _to_ms(record['repair_time']),
                        'repair_solve_ms': _to_ms(record['repair_solve_time']),
                        'repair_prepare_ms': _to_ms(record['repair_prepare_time']),
                        'repair_verify_ms': _to_ms(record['repair_verify_time']),
                        'compute_total_ms': _to_ms(record['compute_time']),
                    }
                )

    aggregate = _aggregate_run_summaries(run_summaries)
    table_tex = _make_table_tex(args, aggregate)

    if args.save_csv is not None:
        _write_raw_csv(args.save_csv, raw_rows)
    if args.save_summary_json is not None:
        ensure_parent_dir(args.save_summary_json)
        with open(args.save_summary_json, 'w', encoding='ascii') as fh:
            json.dump(
                {
                    'config': {
                        'seed': args.seed,
                        'runs': args.runs,
                        'warmup_runs': args.warmup_runs,
                        'course_length': args.course_length,
                        'road_half_width': args.road_half_width,
                        'stages': args.stages,
                        'v0': args.v0,
                        'max_steps': args.max_steps,
                        'goal_tol': args.goal_tol,
                        'speed_tol': args.speed_tol,
                        't_move': args.t_move,
                        'verify_uncertainty': args.verify_uncertainty,
                        'verify_dt': args.verify_dt,
                        'repair_max_iters': args.repair_max_iters,
                        'repair_speed_backoff': args.repair_speed_backoff,
                        'repair_buffer_step': args.repair_buffer_step,
                        'obstacle_inset': args.obstacle_inset,
                    },
                    'run_summaries': run_summaries,
                    'aggregate': aggregate,
                },
                fh,
                indent=2,
            )
            fh.write('\n')
    if args.save_table_tex is not None:
        ensure_parent_dir(args.save_table_tex)
        with open(args.save_table_tex, 'w', encoding='ascii') as fh:
            fh.write(table_tex)

    print(
        f'Case Study 3 timing benchmark: seed={args.seed}, warmup_runs={args.warmup_runs}, '
        f'runs={args.runs}'
    )
    for planner in ('standard', 'rtd_rax'):
        metrics = aggregate[planner]['metrics']
        total = metrics['compute_time']
        print(
            f"  {planner}: total cycle = {total['mean_ms']:.2f} +/- {total['std_ms']:.2f} ms, "
            f"runs={aggregate[planner]['runs']}"
        )


if __name__ == '__main__':
    main()
