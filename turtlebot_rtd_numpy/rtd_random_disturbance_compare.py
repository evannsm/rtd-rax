"""
Random-disturbance compare case study.

This script builds a short random multi-gap course with disturbance patches,
then compares:
  - standard RTD executed directly under disturbances
  - RTD-RAX (noerror FRS + immrax verification/repair using measured disturbance)
"""

import os
import argparse
import matplotlib

if 'MPLBACKEND' not in os.environ:
    if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
        matplotlib.use('TkAgg')
    else:
        matplotlib.use('Agg')

import matplotlib.pyplot as plt

from disturbance_case_study_utils import (
    animate_compare_episodes,
    generate_gap_patch_course,
    inset_road_edge_obstacles,
    load_case_study_models,
    plot_compare_episodes,
    print_result_summary,
    run_episode,
)
from immrax_verify import warmup_verifier


def _parse_args():
    p = argparse.ArgumentParser(description='Random disturbance compare case study')
    p.add_argument('--seed', type=int, default=32)
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
    p.add_argument('--save-fig', type=str, default=None)
    p.add_argument('--save-animation', type=str, default=None)
    p.add_argument('--animation-fps', type=int, default=12)
    p.add_argument('--title', type=str, default='Planning Against Disturbances')
    p.add_argument('--no-show', action='store_true')
    return p.parse_args()


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
        title=args.title,
    )
    scenario = inset_road_edge_obstacles(scenario, args.obstacle_inset)
    models = load_case_study_models()

    standard_result = run_episode(
        scenario,
        'standard',
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
    rax_result = run_episode(
        scenario,
        'rtd_rax',
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

    print_result_summary(standard_result)
    print_result_summary(rax_result)

    plot_compare_episodes(scenario, standard_result, rax_result, save_path=args.save_fig)
    if args.save_animation is not None:
        animate_compare_episodes(
            scenario,
            standard_result,
            rax_result,
            save_path=args.save_animation,
            fps=args.animation_fps,
        )

    if args.no_show:
        plt.close('all')
    else:
        plt.show()


if __name__ == '__main__':
    main()
