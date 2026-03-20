"""
Case 2 — Angled obstacle animation: Standard RTD vs Non-Inflated RTD vs RTD-RAX
================================================================================

Three-panel comparison showing:
  - Standard RTD: conservative, takes a wide path around obstacle (safe)
  - Non-Inflated RTD: tighter path, but can fail to make progress under the same disturbance
  - RTD-RAX: tighter path with immrax verification + repair (safe)

The Non-Inflated FRS lacks the tracking-error buffer, so its planned paths
cut closer to the obstacle. RTD-RAX uses the same Non-Inflated FRS but
verifies each step with immrax and repairs unsafe candidates online.
"""

import os
import json
import argparse
import numpy as np
import matplotlib

if 'MPLBACKEND' not in os.environ:
    if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
        matplotlib.use('TkAgg')
    else:
        matplotlib.use('Agg')

import matplotlib.pyplot as plt

from disturbance_case_study_utils import (
    DisturbancePatch,
    Scenario,
    animate_triple_compare_episodes,
    animate_rax_repair_view,
    load_case_study_models,
    make_rect_polygon,
    plot_rax_repair_view,
    plot_triple_compare_episodes,
    print_result_summary,
    run_episode,
)
from immrax_verify import warmup_verifier


def _parse_args():
    p = argparse.ArgumentParser(
        description='Case 2: Angled obstacle animation — Standard vs Non-Inflated vs RTD-RAX',
    )
    p.add_argument('--v0', type=float, default=0.75)
    p.add_argument('--x0', type=float, default=0.0)
    p.add_argument('--y0', type=float, default=0.0)
    p.add_argument('--h0', type=float, default=-1.5707963267948966)  # -pi/2
    p.add_argument('--x-des', type=float, default=1.1261383717579623)
    p.add_argument('--y-des', type=float, default=-1.95)
    p.add_argument('--obs-x', type=float, default=0.3028769328416644)
    p.add_argument('--obs-y', type=float, default=-1.2943691173614533)
    p.add_argument('--obs-half-width', type=float, default=0.14983429713340735)
    p.add_argument('--obs-half-height', type=float, default=0.27546211625061284)
    p.add_argument('--max-steps', type=int, default=26)
    p.add_argument('--goal-tol', type=float, default=0.10)
    p.add_argument('--speed-tol', type=float, default=0.08)
    p.add_argument('--t-move', type=float, default=0.45)
    p.add_argument('--verify-uncertainty', type=float, default=0.03)
    p.add_argument('--verify-dt', type=float, default=0.01)
    p.add_argument('--repair-max-iters', type=int, default=6)
    p.add_argument('--repair-speed-backoff', type=float, default=0.0)
    p.add_argument('--repair-buffer-step', type=float, default=0.015)
    p.add_argument('--repair-push-iters', type=int, default=2,
                   help='Number of lateral repair push attempts per repair iteration (default: 2)')
    p.add_argument('--repair-push-k1-step', type=float, default=0.5,
                   help='Lateral k1 repair step size used when trying push repairs (default: 0.5)')
    p.add_argument('--verify-disturbance', type=float, default=0.03,
                   help='Model-uncertainty disturbance bound for immrax verification (default: 0.03)')
    p.add_argument('--verify-disturbance-components', type=float, nargs=4,
                   default=(0.017, 0.0595, 0.0425, 0.017),
                   metavar=('DX', 'DY', 'DH', 'DV'),
                   help='Optional per-state immrax verification disturbance floors')
    p.add_argument('--baseline-obstacle-buffer', type=float, default=0.08,
                   help='Obstacle buffer used by Standard RTD and Non-Inflated RTD (default: 0.08)')
    p.add_argument('--rax-obstacle-buffer', type=float, default=0.04,
                   help='Obstacle buffer used by RTD-RAX (default: 0.04)')
    p.add_argument('--execution-disturbance', type=float, default=0.03,
                   help='Random per-step execution disturbance bound (model mismatch, default: 0.03)')
    p.add_argument('--execution-disturbance-components', type=float, nargs=4,
                   default=(0.02, 0.10, 0.06, 0.03),
                   metavar=('DX', 'DY', 'DH', 'DV'),
                   help='Optional per-state random execution disturbance bounds')
    p.add_argument('--execution-disturbance-mode', choices=('step', 'episode'), default='episode',
                   help='Use fresh random execution mismatch each step or one random mismatch for the whole episode')
    p.add_argument('--execution-seed', type=int, default=275,
                   help='RNG seed for execution disturbance (default: 275)')
    p.add_argument('--disturbance-dx', type=float, default=0.0,
                   help='Optional disturbance x-component (default: 0.0, no disturbance)')
    p.add_argument('--patch-x-lo', type=float, default=-0.3)
    p.add_argument('--patch-x-hi', type=float, default=0.9)
    p.add_argument('--patch-y-lo', type=float, default=-1.7)
    p.add_argument('--patch-y-hi', type=float, default=-0.2)
    p.add_argument('--save-fig', type=str, default=None,
                   help='Save static figure with legend (PDF)')
    p.add_argument('--save-fig-no-legend', type=str, default=None,
                   help='Save static figure without legend (PDF)')
    p.add_argument('--save-animation', type=str, default=None,
                   help='Save animation (GIF)')
    p.add_argument('--save-summary-json', type=str, default=None,
                   help='Save Case 2 summary metrics, including path lengths (JSON)')
    p.add_argument('--save-repair-view-fig', type=str, default=None,
                   help='Save alternate one-panel repair-view figure with legend (PDF)')
    p.add_argument('--save-repair-view-fig-no-legend', type=str, default=None,
                   help='Save alternate one-panel repair-view figure without legend (PDF)')
    p.add_argument('--save-repair-view-animation', type=str, default=None,
                   help='Save alternate one-panel repair-view animation (GIF)')
    p.add_argument('--animation-fps', type=int, default=12)
    p.add_argument('--no-show', action='store_true')
    return p.parse_args()


def _save_summary_json(save_path, args, standard_result, noerror_result, rax_result):
    if save_path is None:
        return
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    def _result_entry(result):
        return {
            'status': result['status'],
            'steps': int(len(result['step_records'])),
            'repair_count': int(result['repair_count']),
            'goal_distance_final_m': float(result['goal_distance_final']),
            'path_arclength_m': float(result['path_arclength']),
        }

    payload = {
        'scenario': 'Case 2: Angled Obstacle',
        'execution_seed': int(args.execution_seed),
        'execution_disturbance_mode': args.execution_disturbance_mode,
        'execution_disturbance_components': [float(v) for v in args.execution_disturbance_components],
        'verify_disturbance_components': [float(v) for v in args.verify_disturbance_components],
        'baseline_obstacle_buffer_m': float(args.baseline_obstacle_buffer),
        'rax_obstacle_buffer_m': float(args.rax_obstacle_buffer),
        'repair_settings': {
            'max_iters': int(args.repair_max_iters),
            'speed_backoff': float(args.repair_speed_backoff),
            'buffer_step': float(args.repair_buffer_step),
            'push_iters': int(args.repair_push_iters),
            'push_k1_step': float(args.repair_push_k1_step),
        },
        'results': {
            standard_result['label']: _result_entry(standard_result),
            noerror_result['label']: _result_entry(noerror_result),
            rax_result['label']: _result_entry(rax_result),
        },
    }
    with open(save_path, 'w', encoding='ascii') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write('\n')
    print(f"Saved summary JSON: {save_path}")


def _make_angled_obstacle_scenario(args):
    """Build a Scenario for the angled obstacle test with disturbance patch."""
    obs_x_lo = args.obs_x - args.obs_half_width
    obs_x_hi = args.obs_x + args.obs_half_width
    obs_y_lo = args.obs_y - args.obs_half_height
    obs_y_hi = args.obs_y + args.obs_half_height

    obs_poly = make_rect_polygon(obs_x_lo, obs_x_hi, obs_y_lo, obs_y_hi)
    obstacle_rects = [(obs_x_lo, obs_x_hi, obs_y_lo, obs_y_hi)]

    patches = []
    if abs(args.disturbance_dx) > 1e-6:
        patches.append(DisturbancePatch(
            rect=(args.patch_x_lo, args.patch_x_hi, args.patch_y_lo, args.patch_y_hi),
            disturbance=np.array([args.disturbance_dx, 0.0, 0.0, 0.0]),
            label='Ice patch',
        ))

    return Scenario(
        name='Angled Obstacle',
        start_pose=np.array([args.x0, args.y0, args.h0]),
        goal=np.array([[args.x_des], [args.y_des]]),
        road_half_width=None,
        world_bounds=None,
        obstacle_polys=[obs_poly],
        obstacle_rects=obstacle_rects,
        patches=patches,
        seed=0,
    )


def _resolve_disturbance_arg(scalar_value, component_values):
    if component_values is None:
        return scalar_value
    return np.asarray(component_values, dtype=float)


def main():
    args = _parse_args()
    warmup_verifier(dt=args.verify_dt)

    scenario = _make_angled_obstacle_scenario(args)
    models = load_case_study_models()

    common_kwargs = dict(
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
        repair_push_iters=args.repair_push_iters,
        repair_push_k1_step=args.repair_push_k1_step,
        verify_disturbance=_resolve_disturbance_arg(
            args.verify_disturbance,
            args.verify_disturbance_components,
        ),
        execution_disturbance=_resolve_disturbance_arg(
            args.execution_disturbance,
            args.execution_disturbance_components,
        ),
        execution_disturbance_seed=args.execution_seed,
        execution_disturbance_mode=args.execution_disturbance_mode,
    )

    baseline_obstacle_buffer = args.baseline_obstacle_buffer
    rax_obstacle_buffer = args.rax_obstacle_buffer
    if rax_obstacle_buffer is None:
        rax_obstacle_buffer = baseline_obstacle_buffer

    standard_kwargs = dict(common_kwargs)
    noerror_kwargs = dict(common_kwargs)
    rax_kwargs = dict(common_kwargs)
    if baseline_obstacle_buffer is not None:
        standard_kwargs['obstacle_buffer'] = baseline_obstacle_buffer
        noerror_kwargs['obstacle_buffer'] = baseline_obstacle_buffer
    if rax_obstacle_buffer is not None:
        rax_kwargs['obstacle_buffer'] = rax_obstacle_buffer

    standard_result = run_episode(
        scenario,
        'standard',
        models,
        **standard_kwargs,
    )
    noerror_result = run_episode(
        scenario,
        'noerror',
        models,
        **noerror_kwargs,
    )
    rax_result = run_episode(
        scenario,
        'rtd_rax',
        models,
        store_verify_results=True,
        **rax_kwargs,
    )

    print_result_summary(standard_result)
    print_result_summary(noerror_result)
    print_result_summary(rax_result)
    _save_summary_json(
        args.save_summary_json,
        args,
        standard_result,
        noerror_result,
        rax_result,
    )

    if args.save_fig:
        plot_triple_compare_episodes(
            scenario, standard_result, noerror_result, rax_result,
            models=models,
            save_path=args.save_fig, show_legend=True,
        )
    if args.save_fig_no_legend:
        plot_triple_compare_episodes(
            scenario, standard_result, noerror_result, rax_result,
            models=models,
            save_path=args.save_fig_no_legend, show_legend=False,
        )
    if args.save_animation is not None:
        animate_triple_compare_episodes(
            scenario, standard_result, noerror_result, rax_result,
            models=models,
            save_path=args.save_animation,
            fps=args.animation_fps,
        )
    if args.save_repair_view_fig:
        plot_rax_repair_view(
            scenario,
            noerror_result,
            rax_result,
            models,
            save_path=args.save_repair_view_fig,
            show_legend=True,
        )
    if args.save_repair_view_fig_no_legend:
        plot_rax_repair_view(
            scenario,
            noerror_result,
            rax_result,
            models,
            save_path=args.save_repair_view_fig_no_legend,
            show_legend=False,
        )
    if args.save_repair_view_animation is not None:
        animate_rax_repair_view(
            scenario,
            noerror_result,
            rax_result,
            models,
            save_path=args.save_repair_view_animation,
            fps=args.animation_fps,
            show_legend=False,
        )

    if (
        not args.save_fig
        and not args.save_fig_no_legend
        and not args.save_animation
        and not args.save_repair_view_fig
        and not args.save_repair_view_fig_no_legend
        and not args.save_repair_view_animation
    ):
        plot_triple_compare_episodes(
            scenario,
            standard_result,
            noerror_result,
            rax_result,
            models=models,
        )

    if args.no_show:
        plt.close('all')
    else:
        plt.show()


if __name__ == '__main__':
    main()
