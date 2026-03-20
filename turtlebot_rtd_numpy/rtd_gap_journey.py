"""
rtd_gap_journey.py  –  NumPy / SciPy version
==============================================
Receding-horizon RTD through the fixed two-obstacle gap until the goal is reached.

This extends the one-shot demo by repeatedly:
  1) mapping the current obstacle geometry to the current FRS frame
  2) solving for k_opt
  3) executing only a short prefix of the braking trajectory (t_move)
  4) replanning from the new state
"""

import os
import sys
import argparse
import numpy as np
import matplotlib

if 'MPLBACKEND' not in os.environ:
    if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
        matplotlib.use('TkAgg')
    else:
        matplotlib.use('Agg')

import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds

sys.path.insert(0, os.path.dirname(__file__))

from frs_loader import load_frs, k_to_wv, _DEFAULT_DIR
from geometry_utils import (
    world_to_local, FRS_to_world,
    compute_turtlebot_point_spacing, compute_turtlebot_discretized_obs,
)
from trajectory import make_turtlebot_braking_trajectory
from turtlebot_agent import TurtlebotAgent
from polynomial_utils import (
    get_frs_polynomial_structure,
    evaluate_frs_polynomial_on_obstacle_points,
    get_constraint_polynomial_gradient,
    eval_constraint_poly,
)
from cost import turtlebot_cost_and_grad
from constraints import build_constraint


def _parse_args():
    p = argparse.ArgumentParser(description='Multi-step RTD journey through gap scenario')
    p.add_argument('--frs', choices=['standard', 'noerror'], default='noerror')
    p.add_argument('--max-steps', type=int, default=60,
                   help='Maximum replanning iterations (default: 60)')
    p.add_argument('--goal-tol', type=float, default=0.10,
                   help='Goal distance tolerance in meters (default: 0.10)')
    p.add_argument('--speed-tol', type=float, default=0.08,
                   help='Speed tolerance for goal completion (default: 0.08 m/s)')
    p.add_argument('--t-move', type=float, default=0.5,
                   help='Execution time per replan step (default: 0.5 s)')
    p.add_argument('--v0', type=float, default=0.75,
                   help='Initial speed (default: 0.75 m/s)')
    p.add_argument('--x-des', type=float, default=2.0,
                   help='Goal x in world frame (default: 2.0)')
    p.add_argument('--y-des', type=float, default=0.0,
                   help='Goal y in world frame (default: 0.0)')
    p.add_argument('--no-show', action='store_true',
                   help='Do not call plt.show() (useful for non-interactive runs)')
    p.add_argument('--verify', action='store_true',
                   help='Run immrax verification during the journey')
    p.add_argument('--verify-every', type=int, default=1,
                   help='Run immrax every N replans (default: 1 = every step)')
    p.add_argument('--verify-uncertainty', type=float, default=0.01,
                   help='Additional positional uncertainty for immrax during journey (default: 0.01)')
    p.add_argument('--verify-disturbance', type=float, default=0.0,
                   help='Bounded disturbance for immrax during journey (default: 0.0)')
    return p.parse_args()


# Gap scenario constants
OBS_X = 0.75
OBS_HALF_WIDTH = 0.4
GAP_WIDTH = 0.619
OBS_HEIGHT = 0.6
OBSTACLE_BUFFER = 0.05

_FRS_PATHS = {
    'standard': os.path.join(_DEFAULT_DIR, 'turtlebot_FRS_deg_10_v_0_0.5_to_1.0_preproc.mat'),
    'noerror': os.path.join(_DEFAULT_DIR, 'turtlebot_FRS_deg_10_v_0_0.5_to_1.0_noerror_preproc.mat'),
}


def make_rect_polygon(x_lo, x_hi, y_lo, y_hi):
    xs = [x_lo, x_hi, x_hi, x_lo, x_lo]
    ys = [y_lo, y_lo, y_hi, y_hi, y_lo]
    return np.array([xs, ys], dtype=float)


def _truncate_reference(T, U, Z, t_exec):
    """Truncate reference trajectory to [0, t_exec], interpolating the endpoint."""
    t_exec = float(np.clip(t_exec, 0.0, float(T[-1])))
    if t_exec >= float(T[-1]) - 1e-12:
        return T, U, Z

    keep = T < t_exec
    T_new = np.concatenate([T[keep], np.array([t_exec])])
    U_new = np.vstack([np.interp(T_new, T, U[i, :]) for i in range(U.shape[0])])
    Z_new = np.vstack([np.interp(T_new, T, Z[i, :]) for i in range(Z.shape[0])])
    return T_new, U_new, Z_new


def _compute_frs_contour(frs, k_opt, initial_pose, grid_res=120):
    z1g, z2g = np.meshgrid(np.linspace(-1, 1, grid_res), np.linspace(-1, 1, grid_res))
    z_grid = np.vstack([z1g.ravel(), z2g.ravel()])

    pows = frs['pows']
    coef = frs['coef']
    z_cols = frs['z_cols']
    k_cols = frs['k_cols']

    k_pows = pows[:, k_cols]
    z_pows = pows[:, z_cols]
    k_mono = np.prod(k_opt[np.newaxis, :] ** k_pows, axis=1)
    z_vals = np.prod(z_grid[np.newaxis, :, :] ** z_pows[:, :, np.newaxis], axis=1)
    frs_grid = (coef * k_mono) @ z_vals

    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(z1g, z2g, frs_grid.reshape(z1g.shape), levels=[0.0])
    plt.close(fig_tmp)

    if not cs.allsegs or not cs.allsegs[0]:
        return None

    segs = [s for s in cs.allsegs[0] if s.size > 0]
    if not segs:
        return None

    parts = []
    for i, seg in enumerate(segs):
        parts.append(seg.T)
        if i < len(segs) - 1:
            parts.append(np.full((2, 1), np.nan))
    c_frs = np.hstack(parts)

    x0, y0 = frs['initial_x'], frs['initial_y']
    d = frs['distance_scale']
    return FRS_to_world(c_frs, initial_pose, x0, y0, d)


def main():
    args = _parse_args()
    if args.verify_every < 1:
        raise ValueError('--verify-every must be >= 1')

    frs = load_frs(path=_FRS_PATHS[args.frs])
    agent = TurtlebotAgent()
    agent.reset([0.0, 0.0, 0.0, args.v0])
    immrax_verify = None
    if args.verify:
        from immrax_verify import verify as immrax_verify

    x_lo = OBS_X - OBS_HALF_WIDTH
    x_hi = OBS_X + OBS_HALF_WIDTH
    half_gap = GAP_WIDTH / 2.0
    o_upper = make_rect_polygon(x_lo, x_hi, half_gap, half_gap + OBS_HEIGHT)
    o_lower = make_rect_polygon(x_lo, x_hi, -half_gap - OBS_HEIGHT, -half_gap)
    obstacle_rects = [
        (x_lo, x_hi, half_gap, half_gap + OBS_HEIGHT),
        (x_lo, x_hi, -half_gap - OBS_HEIGHT, -half_gap),
    ]

    goal_world = np.array([[args.x_des], [args.y_des]], dtype=float)
    r = compute_turtlebot_point_spacing(agent.footprint, OBSTACLE_BUFFER)
    _, o_buf_upper, o_pts_upper = compute_turtlebot_discretized_obs(
        o_upper, agent.state[:, -1], OBSTACLE_BUFFER, r, frs
    )
    _, o_buf_lower, o_pts_lower = compute_turtlebot_discretized_obs(
        o_lower, agent.state[:, -1], OBSTACLE_BUFFER, r, frs
    )
    fp = get_frs_polynomial_structure(frs['pows'], frs['coef'], frs['z_cols'], frs['k_cols'])

    k_hist = []
    plan_pose_hist = []
    contour_world_hist = []
    verify_steps = []
    verify_safe = []
    verify_traces = []
    status = 'max_steps_reached'

    print(f"Starting journey with FRS='{args.frs}', goal=({args.x_des:.2f}, {args.y_des:.2f})")

    for step in range(args.max_steps):
        pose = agent.pose
        speed = agent.speed
        dist_goal = float(np.linalg.norm(pose[:2] - goal_world.ravel()))
        print(f'[step {step:02d}] pose=({pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f}), v={speed:.3f}, d_goal={dist_goal:.3f}')

        if dist_goal <= args.goal_tol and speed <= args.speed_tol:
            status = 'goal_reached'
            break

        # Goal in current local frame
        z_goal_local = np.asarray(world_to_local(agent.state[:, -1], goal_world)).reshape(-1)
        x_des_loc = float(z_goal_local[0])
        y_des_loc = float(z_goal_local[1])

        # Obstacle discretization in current FRS frame
        o_frs_parts = []
        for o in (o_upper, o_lower):
            o_frs_i, _, _ = compute_turtlebot_discretized_obs(
                o, agent.state[:, -1], OBSTACLE_BUFFER, r, frs
            )
            if o_frs_i.shape[1] > 0:
                o_frs_parts.append(o_frs_i)
        o_frs = np.hstack(o_frs_parts) if o_frs_parts else np.zeros((2, 0))

        # Build constraints
        if o_frs.shape[1] > 0:
            cons_poly = evaluate_frs_polynomial_on_obstacle_points(fp, o_frs)
            cons_grad = get_constraint_polynomial_gradient(cons_poly)
            constraint = build_constraint(cons_poly, cons_grad)
            constraints_list = [constraint]
        else:
            cons_poly = None
            constraints_list = []

        # Bounds around current speed
        v_cur = float(np.clip(speed, frs['v_range'][0], frs['v_range'][1]))
        v_des_lo = max(v_cur - frs['delta_v'], frs['v_range'][0])
        v_des_hi = min(v_cur + frs['delta_v'], frs['v_range'][1])
        v_max = frs['v_range'][1]
        k2_lo = (v_des_lo - v_max / 2.0) * (2.0 / v_max)
        k2_hi = (v_des_hi - v_max / 2.0) * (2.0 / v_max)
        bounds = Bounds(lb=[-1.0, k2_lo], ub=[1.0, k2_hi])

        x0 = k_hist[-1] if k_hist else np.zeros(2)
        result = minimize(
            fun=lambda k: turtlebot_cost_and_grad(k, frs['w_max'], v_max, x_des_loc, y_des_loc),
            x0=x0,
            jac=True,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 200, 'ftol': 1e-6, 'disp': False},
        )

        if not (result.success or result.status == 0):
            status = 'infeasible'
            print(f'  no feasible k ({result.message})')
            if speed > args.speed_tol:
                t_stop = speed / agent.max_accel
                t_brk, u_brk, z_brk = make_turtlebot_braking_trajectory(0.0, t_stop, 0.0, speed)
                agent.move(t_brk[-1], t_brk, u_brk, z_brk)
                status = 'braked_after_infeasible'
                print(f'  executed emergency brake for {t_brk[-1]:.3f}s')
            break

        k_opt = result.x
        k_hist.append(k_opt)
        plan_pose_hist.append(agent.state[:3, -1].copy())

        if cons_poly is not None:
            g = eval_constraint_poly(cons_poly, k_opt)
            print(f'  k_opt={k_opt}, max g(k_opt)={np.max(g):.3e}')
        else:
            print(f'  k_opt={k_opt}, no active obstacle constraints')

        # Plan full braking trajectory but execute only t_move (RTD receding horizon)
        w_des, v_des = k_to_wv(k_opt, frs)
        t_plan = frs['t_plan']
        t_stop = v_des / agent.max_accel

        if args.verify and (step % args.verify_every == 0):
            vres = immrax_verify(
                w_des=w_des,
                v_des=v_des,
                t_plan=t_plan,
                t_stop=t_stop,
                z0=agent.state[:, -1].tolist(),
                obstacle_rects=obstacle_rects,
                robot_radius=agent.footprint,
                obstacle_inflate_radius=0.0,
                init_uncertainty=args.verify_uncertainty,
                disturbance_bound=args.verify_disturbance,
            )
            verify_steps.append(step)
            verify_safe.append(bool(vres['safe']))
            verify_traces.append({
                'step': int(step),
                'safe': bool(vres['safe']),
                'xy_tube': np.asarray(vres.get('xy_tube', np.zeros((0, 4))), dtype=float),
                'nom_xy': np.asarray(vres.get('nom_xy', np.zeros((0, 2))), dtype=float),
            })
            vtxt = 'SAFE' if vres['safe'] else 'COLLISION'
            print(f'  immrax(step {step:02d}) -> {vtxt}')

        t_ref, u_ref, z_ref = make_turtlebot_braking_trajectory(t_plan, t_stop, w_des, v_des)
        t_exec = min(float(args.t_move), float(t_ref[-1]))
        t_exec_ref, u_exec, z_exec = _truncate_reference(t_ref, u_ref, z_ref, t_exec)

        n_prev = agent.state.shape[1]
        agent.move(t_exec_ref[-1], t_exec_ref, u_exec, z_exec)

        # Goal detection over the executed trajectory segment (not only at step endpoints).
        seg_xy = agent.state[:2, n_prev - 1:]  # include segment start + new samples
        seg_dist = np.linalg.norm(seg_xy - goal_world, axis=0)
        hit_idx = np.where(seg_dist <= args.goal_tol)[0]
        if hit_idx.size > 0:
            # Trim history at first goal-contact sample to avoid overshoot in logs/plots.
            gidx = (n_prev - 1) + int(hit_idx[0])
            agent.state = agent.state[:, :gidx + 1]
            agent.time = agent.time[:gidx + 1]
            status = 'goal_reached'
            print(f'  goal reached during step at t={agent.time[-1]:.3f}s, d_goal={seg_dist[hit_idx[0]]:.3f}')
            break

        c_world = _compute_frs_contour(frs, k_opt, plan_pose_hist[-1], grid_res=90)
        if c_world is not None:
            contour_world_hist.append(c_world)

    # Final summary
    final_pose = agent.pose
    final_dist = float(np.linalg.norm(final_pose[:2] - goal_world.ravel()))
    print(f'Finished with status={status}, final_pose={final_pose}, d_goal={final_dist:.3f}')

    _plot_journey(
        agent=agent,
        goal_world=goal_world.ravel(),
        o_upper=o_upper,
        o_lower=o_lower,
        o_buf_upper=o_buf_upper,
        o_buf_lower=o_buf_lower,
        o_pts_upper=o_pts_upper,
        o_pts_lower=o_pts_lower,
        k_hist=np.array(k_hist) if k_hist else np.zeros((0, 2)),
        plan_pose_hist=np.array(plan_pose_hist) if plan_pose_hist else np.zeros((0, 3)),
        contour_world_hist=contour_world_hist,
        verify_steps=np.array(verify_steps, dtype=int),
        verify_safe=np.array(verify_safe, dtype=bool),
        verify_traces=verify_traces,
        status=status,
        frs_name=args.frs,
        goal_tol=args.goal_tol,
    )
    plt.tight_layout()
    if args.no_show:
        plt.close('all')
    else:
        plt.show()


def _plot_journey(agent, goal_world, o_upper, o_lower, o_buf_upper, o_buf_lower, o_pts_upper, o_pts_lower,
                  k_hist, plan_pose_hist,
                  contour_world_hist, verify_steps, verify_safe, verify_traces,
                  status, frs_name, goal_tol):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    ax = axes[0]
    ax.set_title(f'Gap Journey ({frs_name})  |  status: {status}')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Obstacles with RTD-style buffered boundary + sampled points.
    obs_color = [0.75, 0.25, 0.25]
    buf_color = [1.0, 0.55, 0.55]
    pts_color = [0.4, 0.05, 0.05]
    for o_raw, o_buf, o_pts in ((o_upper, o_buf_upper, o_pts_upper), (o_lower, o_buf_lower, o_pts_lower)):
        ax.fill(o_raw[0], o_raw[1], color=obs_color, alpha=0.9, label='_nolegend_')
        if o_buf is not None and o_buf.shape[1] > 0:
            ax.fill(o_buf[0], o_buf[1], color=buf_color, alpha=0.5, label='_nolegend_')
        if o_pts is not None and o_pts.shape[1] > 0:
            ax.plot(o_pts[0], o_pts[1], '.', color=pts_color, markersize=3, label='_nolegend_')
    ax.fill([], [], color=obs_color, alpha=0.9, label='obstacle')
    ax.fill([], [], color=buf_color, alpha=0.5, label='buffered obstacle')
    ax.plot([], [], '.', color=pts_color, markersize=4, label='obs pts')

    # Planned FRS contours from each step
    for c_world in contour_world_hist:
        ax.plot(c_world[0], c_world[1], color=[0.25, 0.8, 0.35], alpha=0.18, linewidth=1.0)
    if contour_world_hist:
        ax.plot([], [], color=[0.25, 0.8, 0.35], alpha=0.6, linewidth=1.2, label='FRS @ replans')

    # Journey path
    st = agent.state
    ax.plot(st[0, :], st[1, :], color='purple', linestyle='--', linewidth=2.0, label='RTD path')
    ax.plot(st[0, 0], st[1, 0], 'ko', markerfacecolor='none', label='start')
    ax.plot(st[0, -1], st[1, -1], 'ko', markersize=5, label='final')
    ax.plot(goal_world[0], goal_world[1], 'k*', markersize=14, label='goal')

    goal_circle = plt.Circle((goal_world[0], goal_world[1]), goal_tol, color='k', fill=False,
                             linestyle=':', linewidth=1.2, alpha=0.8)
    ax.add_patch(goal_circle)

    if plan_pose_hist.shape[0] > 0:
        ax.plot(plan_pose_hist[:, 0], plan_pose_hist[:, 1], '.', color='black', markersize=4,
                label='replan points')

    # Compact immrax tube visualization: evenly sample traces across the whole
    # journey (always includes first and last) so there are no large visual gaps.
    max_traces = 7
    if len(verify_traces) > max_traces:
        idx = np.linspace(0, len(verify_traces) - 1, num=max_traces, dtype=int)
        idx = np.unique(idx)
        traces_to_plot = [verify_traces[i] for i in idx]
    else:
        traces_to_plot = verify_traces
    for i, tr in enumerate(traces_to_plot):
        xy_tube = tr['xy_tube']
        nom_xy = tr['nom_xy']
        if xy_tube.shape[0] == 0:
            continue
        safe = tr['safe']
        base_col = np.array([0.15, 0.45, 0.95]) if safe else np.array([0.90, 0.20, 0.20])
        fade = 0.45 + 0.50 * (i + 1) / max(len(traces_to_plot), 1)
        col = tuple(np.clip(base_col * fade, 0.0, 1.0))

        x_mid = 0.5 * (xy_tube[:, 0] + xy_tube[:, 1])
        y_lo = xy_tube[:, 2]
        y_hi = xy_tube[:, 3]
        ax.plot(x_mid, y_lo, color=col, linewidth=1.2, alpha=0.75, zorder=5)
        ax.plot(x_mid, y_hi, color=col, linewidth=1.2, alpha=0.75, zorder=5)
        if nom_xy.shape[0] > 0:
            ax.plot(nom_xy[:, 0], nom_xy[:, 1], color=col, linestyle=':', linewidth=1.3,
                    alpha=0.95, zorder=6)

    if len(traces_to_plot) > 0:
        ax.plot([], [], color=[0.15, 0.45, 0.95], linewidth=1.2, label='Immrax tube bounds (SAFE)')
        ax.plot([], [], color=[0.90, 0.20, 0.20], linewidth=1.2, label='Immrax tube bounds (COLLISION)')
        ax.plot([], [], color='gray', linestyle=':', linewidth=1.2, label='Immrax nominal (sampled)')

    pad_x = 0.3
    pad_y = 0.25
    x_all = np.r_[st[0, :], o_upper[0, :], o_lower[0, :], goal_world[0]]
    y_all = np.r_[st[1, :], o_upper[1, :], o_lower[1, :], goal_world[1]]
    ax.set_xlim(np.min(x_all) - pad_x, np.max(x_all) + pad_x)
    ax.set_ylim(np.min(y_all) - pad_y, np.max(y_all) + pad_y)
    ax.legend(fontsize=8, loc='best')

    axk = axes[1]
    axk.set_title('k History by Replan Step')
    axk.set_xlabel('replan step')
    axk.set_ylabel('k value')
    axk.grid(True, alpha=0.3)
    if k_hist.shape[0] > 0:
        idx = np.arange(k_hist.shape[0])
        axk.plot(idx, k_hist[:, 0], '-o', markersize=3, label='k1 (yaw-rate)')
        axk.plot(idx, k_hist[:, 1], '-o', markersize=3, label='k2 (speed)')
        if verify_steps.size > 0:
            safe_steps = verify_steps[verify_safe]
            unsafe_steps = verify_steps[~verify_safe]
            if safe_steps.size > 0:
                axk.plot(safe_steps, np.zeros_like(safe_steps), '^', color='green',
                         markersize=6, label='Immrax SAFE')
            if unsafe_steps.size > 0:
                axk.plot(unsafe_steps, np.zeros_like(unsafe_steps), 'x', color='red',
                         markersize=6, label='Immrax COLLISION')
        axk.legend(fontsize=8, loc='best')
    else:
        axk.text(0.5, 0.5, 'no feasible replans', transform=axk.transAxes,
                 ha='center', va='center')


if __name__ == '__main__':
    main()
