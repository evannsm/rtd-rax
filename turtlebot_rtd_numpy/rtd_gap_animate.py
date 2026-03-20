"""
Case 1 — Gap animation: Standard RTD vs RTD-RAX
================================================

Two rectangular obstacles form a narrow gap.  Standard RTD (tracking-error FRS)
cannot find a feasible path through the gap, while RTD-RAX (noerror FRS + immrax
verification) fits through safely.

Produces:
  - Side-by-side animation (GIF) with buffered obstacles, discretized points,
    and per-step FRS contours
  - Static figures (PDF) with and without legend
"""

import os
import argparse
import numpy as np
import matplotlib

if 'MPLBACKEND' not in os.environ:
    if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
        matplotlib.use('TkAgg')
    else:
        matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

from disturbance_case_study_utils import (
    CASE_STUDY_STYLE,
    DEFAULT_OBSTACLE_BUFFER,
    Scenario,
    _COMPARE_COLORS,
    apply_case_study_style,
    compute_frs_contour_world,
    compute_step_contours,
    ensure_parent_dir,
    load_case_study_models,
    make_rect_polygon,
    plot_compare_episodes,
    print_result_summary,
    run_episode,
)
from geometry_utils import compute_turtlebot_discretized_obs, compute_turtlebot_point_spacing
from immrax_verify import warmup_verifier

# Gap scenario parameters (match one_shot_rtd_gap.py)
V_0 = 0.75
X_DES = 2.0
Y_DES = 0.0
OBS_X = 0.75
OBS_HALF_WIDTH = 0.4
GAP_WIDTH = 0.619
OBS_HEIGHT = 0.6


def _parse_args():
    p = argparse.ArgumentParser(description='Case 1: Gap animation — Standard RTD vs RTD-RAX')
    p.add_argument('--v0', type=float, default=V_0)
    p.add_argument('--max-steps', type=int, default=30)
    p.add_argument('--goal-tol', type=float, default=0.12)
    p.add_argument('--speed-tol', type=float, default=0.08)
    p.add_argument('--t-move', type=float, default=0.48)
    p.add_argument('--verify-uncertainty', type=float, default=0.01)
    p.add_argument('--verify-dt', type=float, default=0.01)
    p.add_argument('--repair-max-iters', type=int, default=4)
    p.add_argument('--repair-speed-backoff', type=float, default=0.08)
    p.add_argument('--repair-buffer-step', type=float, default=0.015)
    p.add_argument('--save-fig', type=str, default=None,
                   help='Save static figure with legend (PDF)')
    p.add_argument('--save-fig-no-legend', type=str, default=None,
                   help='Save static figure without legend (PDF)')
    p.add_argument('--save-animation', type=str, default=None,
                   help='Save animation (GIF)')
    p.add_argument('--animation-fps', type=int, default=12)
    p.add_argument('--no-show', action='store_true')
    return p.parse_args()


def _make_gap_scenario():
    """Build a Scenario for the two-obstacle gap test."""
    half_gap = GAP_WIDTH / 2.0
    x_lo = OBS_X - OBS_HALF_WIDTH
    x_hi = OBS_X + OBS_HALF_WIDTH

    O_upper = make_rect_polygon(x_lo, x_hi, half_gap, half_gap + OBS_HEIGHT)
    O_lower = make_rect_polygon(x_lo, x_hi, -half_gap - OBS_HEIGHT, -half_gap)

    obstacle_polys = [O_upper, O_lower]
    obstacle_rects = [
        (x_lo, x_hi, half_gap, half_gap + OBS_HEIGHT),
        (x_lo, x_hi, -half_gap - OBS_HEIGHT, -half_gap),
    ]

    return Scenario(
        name='Gap',
        start_pose=np.array([0.0, 0.0, 0.0]),
        goal=np.array([[X_DES], [Y_DES]]),
        road_half_width=None,
        world_bounds=None,
        obstacle_polys=obstacle_polys,
        obstacle_rects=obstacle_rects,
        patches=[],
        seed=0,
    )


# ── Rich animation ─────────────────────────────────────────────────────

_OBS_COLOR = [0.75, 0.25, 0.25]
_BUF_COLOR = [1.0, 0.55, 0.55]
_PTS_COLOR = [0.4, 0.05, 0.05]
_FRS_COLOR = [0.3, 0.8, 0.5]


def _precompute_obs_display(scenario, frs, initial_pose, footprint):
    """Compute buffered obstacle polygons and discretized points (world frame)."""
    r = compute_turtlebot_point_spacing(footprint, DEFAULT_OBSTACLE_BUFFER)
    all_buf = []
    all_pts = []
    for poly in scenario.obstacle_polys:
        _, O_buf, O_pts = compute_turtlebot_discretized_obs(
            poly, initial_pose, DEFAULT_OBSTACLE_BUFFER, r, frs,
        )
        all_buf.append(O_buf)
        all_pts.append(O_pts)
    return all_buf, all_pts


def _build_frame_to_step(result):
    """Map each agent.state frame index to a step index (-1 if before any step)."""
    n_frames = result['agent'].state.shape[1]
    boundaries = []
    for i, rec in enumerate(result['step_records']):
        if 'segment_start' in rec:
            boundaries.append((rec['segment_start'], i))
    mapping = [-1] * n_frames
    for fi in range(n_frames):
        for seg_start, si in boundaries:
            if seg_start <= fi:
                mapping[fi] = si
    return mapping


def _draw_rich_static(ax, scenario, obs_bufs, obs_pts):
    """Draw obstacles with buffer, discretized points, goal, and start."""
    for idx, poly in enumerate(scenario.obstacle_polys):
        poly_n = poly if poly.shape[1] == 5 else np.hstack([poly, poly[:, :1]])
        ax.fill(poly_n[0], poly_n[1], color=_OBS_COLOR, alpha=0.9, zorder=2)

    for O_buf in obs_bufs:
        if O_buf is not None and O_buf.shape[1] > 0:
            ax.fill(O_buf[0], O_buf[1], color=_BUF_COLOR, alpha=0.5, zorder=1)

    for O_pts in obs_pts:
        if O_pts is not None and O_pts.shape[1] > 0:
            ax.plot(O_pts[0], O_pts[1], '.', color=_PTS_COLOR, markersize=5, zorder=3)

    ax.plot(float(scenario.start_pose[0]), float(scenario.start_pose[1]),
            marker='o', markerfacecolor='none', markeredgecolor='black',
            markersize=9, linestyle='None', zorder=5)
    ax.plot(float(scenario.goal[0, 0]), float(scenario.goal[1, 0]),
            'k*', markersize=17, zorder=5)

    half_gap = GAP_WIDTH / 2.0
    y_extent = half_gap + OBS_HEIGHT + 0.15
    x_hi = max(OBS_X + OBS_HALF_WIDTH + 0.3, X_DES + 0.2)
    ax.set_xlim(-0.3, x_hi)
    ax.set_ylim(-y_extent, y_extent)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')


def _make_rich_legend(fig, standard_result, rax_result, anchor):
    """Build a legend matching the one-shot figure style."""
    handles = [
        mpatches.Patch(facecolor=_OBS_COLOR, alpha=0.9, label='Obstacle'),
        mpatches.Patch(facecolor=_BUF_COLOR, alpha=0.5, label='Buffered Obstacle'),
        Line2D([0], [0], marker='.', color=_PTS_COLOR, linestyle='None',
               markersize=5, label='Obstacle Discretization'),
        Line2D([0], [0], marker='*', color='black', linestyle='None',
               markersize=13, label='Goal'),
        Line2D([0], [0], marker='o', markerfacecolor='none', markeredgecolor='black',
               linestyle='None', markersize=9, label='Start'),
    ]
    for r in [standard_result, rax_result]:
        color = _COMPARE_COLORS.get(r['label'], '#333333')
        handles.append(Line2D([0], [0], color=color, linewidth=2.6,
                              label=f"{r['label']} path"))
    handles.append(Line2D([0], [0], color=_FRS_COLOR, linewidth=1.8,
                          label='Nominal FRS'))
    _mmr_cmap = plt.get_cmap('Blues')
    handles.append(Line2D([0], [0], color=_mmr_cmap(0.92), alpha=0.9, linewidth=2.0,
                          label='MMR FRS'))
    if any(r['collision'] is not None for r in [standard_result, rax_result]):
        handles.append(Line2D([0], [0], marker='o', markerfacecolor='none',
                              markeredgecolor='black', markeredgewidth=2.0,
                              color='black', linestyle='None', markersize=12,
                              label='Collision footprint'))
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=anchor,
               frameon=True, borderaxespad=0.0)


def animate_gap_rich(scenario, standard_result, rax_result, models,
                     save_path=None, fps=12, max_frames=240, show_legend=True):
    """Create a rich side-by-side animation with FRS contours and obstacle details."""
    apply_case_study_style()

    frs_standard, _ = models['standard']
    frs_noerror, _ = models['noerror']

    # Precompute obstacle display data
    std_bufs, std_pts = _precompute_obs_display(
        scenario, frs_standard, standard_result['agent'].state[:, 0],
        standard_result['agent'].footprint,
    )
    rax_bufs, rax_pts = _precompute_obs_display(
        scenario, frs_noerror, rax_result['agent'].state[:, 0],
        rax_result['agent'].footprint,
    )

    # Precompute FRS contours for each step
    std_contours = compute_step_contours(standard_result, frs_standard, grid_res=150)
    rax_contours = compute_step_contours(rax_result, frs_noerror, grid_res=150)

    # Build frame→step mapping
    std_f2s = _build_frame_to_step(standard_result)
    rax_f2s = _build_frame_to_step(rax_result)

    fig, axes = plt.subplots(1, 2, figsize=(15.4, 5.8), sharex=True, sharey=True)

    panel_data = [
        (axes[0], standard_result, std_bufs, std_pts, std_contours, std_f2s),
        (axes[1], rax_result, rax_bufs, rax_pts, rax_contours, rax_f2s),
    ]

    # Static green Nominal FRS overlay on Standard RTD panel (from RTD-RAX step 0)
    if rax_contours and rax_contours[0] is not None:
        cw = rax_contours[0]
        axes[0].plot(cw[0], cw[1], color=_FRS_COLOR, linewidth=1.8, zorder=4)

    # Pre-create immrax reach tube box patches for RTD-RAX panel (all invisible)
    _mmr_cmap = plt.get_cmap('Blues')
    mmr_all_rects = []
    mmr_step_ranges = []
    rect_idx = 0
    for rec in rax_result['step_records']:
        vr = rec.get('verify_result')
        if vr is not None:
            xy_tube = np.asarray(vr['xy_tube'])
            n_tube = max(len(xy_tube), 1)
            start = rect_idx
            for i, row in enumerate(xy_tube):
                x_lo, x_hi, y_lo, y_hi = row
                frac = i / max(n_tube - 1, 1)
                col = _mmr_cmap(0.30 + 0.60 * frac)
                rect = mpatches.FancyBboxPatch(
                    (x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                    boxstyle='square,pad=0',
                    linewidth=0, facecolor=col, alpha=0.12,
                    visible=False, zorder=3,
                )
                axes[1].add_patch(rect)
                mmr_all_rects.append(rect)
                rect_idx += 1
            mmr_step_ranges.append((start, rect_idx))
        else:
            mmr_step_ranges.append(None)

    artists = []
    for ax, res, bufs, pts, contours, f2s in panel_data:
        color = _COMPARE_COLORS.get(res['label'], '#333333')
        _draw_rich_static(ax, scenario, bufs, pts)
        ax.set_title(res['label'])

        path_line, = ax.plot([], [], color=color, linewidth=2.6, zorder=6)
        body = mpatches.Circle((0.0, 0.0), res['agent'].footprint,
                               facecolor=color, edgecolor='black', alpha=0.32, zorder=7)
        ax.add_patch(body)
        frs_line, = ax.plot([], [], color=_FRS_COLOR, linewidth=1.8, zorder=4)
        planned_line, = ax.plot([], [], color='purple', linestyle='--',
                                linewidth=2.0, alpha=0.8, zorder=5)
        collision_marker, = ax.plot([], [], marker='x', color='black',
                                    markersize=10, mew=2.2, linestyle='None')
        collision_ring = mpatches.Circle((0.0, 0.0), res['agent'].footprint,
                                         facecolor='none', edgecolor='black',
                                         linewidth=2.2, visible=False, zorder=8)
        ax.add_patch(collision_ring)
        text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top',
                       ha='left', fontsize=11,
                       bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': '0.8'},
                       zorder=10)
        artists.append((res, path_line, body, frs_line, planned_line,
                        collision_marker, collision_ring, text, contours, f2s))

    if show_legend:
        fig.subplots_adjust(left=0.06, right=0.79, bottom=0.12, top=0.92, wspace=0.18)
        right_ax_pos = axes[1].get_position()
        _make_rich_legend(fig, standard_result, rax_result,
                          anchor=(right_ax_pos.x1 + 0.02, right_ax_pos.y1))
    else:
        fig.subplots_adjust(left=0.06, right=0.96, bottom=0.12, top=0.92, wspace=0.18)

    frame_count_raw = max(r['agent'].state.shape[1]
                          for _, r, *_ in panel_data)
    if frame_count_raw <= max_frames:
        frame_indices = np.arange(frame_count_raw, dtype=int)
    else:
        frame_indices = np.unique(
            np.linspace(0, frame_count_raw - 1, int(max_frames), dtype=int))

    def update(frame_idx):
        out = []
        for panel_idx, (res, path_line, body, frs_line, planned_line,
             collision_marker, collision_ring, text, contours, f2s) in enumerate(artists):
            st = res['agent'].state
            idx = min(frame_idx, st.shape[1] - 1)
            path_line.set_data(st[0, :idx + 1], st[1, :idx + 1])
            body.center = (float(st[0, idx]), float(st[1, idx]))

            # FRS contour and planned trajectory for current step
            step_idx = f2s[idx] if idx < len(f2s) else -1
            if 0 <= step_idx < len(contours) and contours[step_idx] is not None:
                cw = contours[step_idx]
                frs_line.set_data(cw[0], cw[1])
            else:
                frs_line.set_data([], [])

            if 0 <= step_idx < len(res['step_records']):
                rec = res['step_records'][step_idx]
                if 'planned_world' in rec:
                    pw = rec['planned_world']
                    planned_line.set_data(pw[0], pw[1])
                else:
                    planned_line.set_data([], [])
            else:
                planned_line.set_data([], [])

            # MMR FRS boxes for RTD-RAX panel (panel_idx == 1)
            if panel_idx == 1 and mmr_all_rects:
                for r in mmr_all_rects:
                    r.set_visible(False)
                if 0 <= step_idx < len(mmr_step_ranges) and mmr_step_ranges[step_idx] is not None:
                    start, end = mmr_step_ranges[step_idx]
                    for r in mmr_all_rects[start:end]:
                        r.set_visible(True)

            # Status text
            is_final = (idx == st.shape[1] - 1)
            if is_final:
                raw_status = res['status']
                if raw_status == 'terminated':
                    status_text = 'fail-safe'
                    text.set_color('red')
                elif raw_status == 'collision':
                    status_text = 'collision'
                    text.set_color('red')
                elif raw_status == 'goal_reached':
                    status_text = 'goal reached'
                    text.set_color('green')
                else:
                    status_text = raw_status
                    text.set_color('black')
            else:
                status_text = 'executing'
                text.set_color('black')

            if res['collision'] is not None and is_final:
                collision_marker.set_data(
                    [res['collision']['point'][0]], [res['collision']['point'][1]])
                collision_ring.center = (
                    float(res['collision']['point'][0]),
                    float(res['collision']['point'][1]))
                collision_ring.set_visible(True)
            else:
                collision_marker.set_data([], [])
                collision_ring.set_visible(False)

            info = (f"t = {res['agent'].time[idx]:.2f} s\n"
                    f"status: {status_text}\n"
                    f"steps: {len(res['step_records'])}")
            if res['repair_count'] > 0:
                info += f" | repairs: {res['repair_count']}"
            text.set_text(info)
            out.extend([path_line, body, frs_line, planned_line,
                        collision_marker, collision_ring, text])
        return out

    anim = FuncAnimation(fig, update, frames=frame_indices,
                         interval=1000.0 / max(float(fps), 1.0), blit=False)
    if save_path is not None:
        ensure_parent_dir(save_path)
        anim.save(save_path, writer=PillowWriter(fps=fps))
        print(f'Saved animation to: {save_path}')
    return fig, anim


def main():
    args = _parse_args()
    warmup_verifier(dt=args.verify_dt)

    scenario = _make_gap_scenario()
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
    )

    standard_result = run_episode(scenario, 'standard', models, **{**common_kwargs, 'max_steps': 1})
    rax_result = run_episode(scenario, 'rtd_rax', models,
                             **{**common_kwargs, 'store_verify_results': True})

    print_result_summary(standard_result)
    print_result_summary(rax_result)

    if args.save_fig:
        plot_compare_episodes(scenario, standard_result, rax_result,
                              save_path=args.save_fig, show_legend=True)
    if args.save_fig_no_legend:
        plot_compare_episodes(scenario, standard_result, rax_result,
                              save_path=args.save_fig_no_legend, show_legend=False)
    if args.save_animation is not None:
        animate_gap_rich(
            scenario, standard_result, rax_result, models,
            save_path=args.save_animation,
            fps=args.animation_fps,
        )

    if not args.save_fig and not args.save_fig_no_legend:
        plot_compare_episodes(scenario, standard_result, rax_result)

    if args.no_show:
        plt.close('all')
    else:
        plt.show()


if __name__ == '__main__':
    main()
