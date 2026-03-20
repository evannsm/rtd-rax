"""
rtd_angled_obstacle_compare.py
==============================
Angled-obstacle case study:
  - execute RTD with noerror FRS
  - solve standard FRS counterfactual at each replan
  - optionally run immrax checks and hybrid repair when unsafe

This scenario starts with the robot facing along -y and places an obstacle on
the down-range path so the planner must choose a turning maneuver around it.
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

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'Nimbus Roman No9 L', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.weight': 'medium',
    'axes.titlesize': 26,
    'axes.labelsize': 25,
    'axes.titleweight': 'medium',
    'axes.labelweight': 'medium',
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 18,
    'figure.titlesize': 27,
    'figure.titleweight': 'medium',
})

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
)
from cost import turtlebot_cost_and_grad
from constraints import build_constraint


_FRS_PATHS = {
    'standard': os.path.join(_DEFAULT_DIR, 'turtlebot_FRS_deg_10_v_0_0.5_to_1.0_preproc.mat'),
    'noerror': os.path.join(_DEFAULT_DIR, 'turtlebot_FRS_deg_10_v_0_0.5_to_1.0_noerror_preproc.mat'),
}


def _parse_args():
    p = argparse.ArgumentParser(description='Angled obstacle compare journey (noerror vs standard + immrax)')
    p.add_argument('--max-steps', type=int, default=70)
    p.add_argument('--goal-tol', type=float, default=0.10)
    p.add_argument('--speed-tol', type=float, default=0.08)
    p.add_argument('--t-move', type=float, default=0.45)
    p.add_argument('--v0', type=float, default=0.75)
    p.add_argument('--x0', type=float, default=0.0)
    p.add_argument('--y0', type=float, default=0.0)
    p.add_argument('--h0', type=float, default=-1.5707963267948966,  # -pi/2
                   help='Initial heading in radians (default: -pi/2)')
    p.add_argument('--x-des', type=float, default=0.95)
    p.add_argument('--y-des', type=float, default=-1.95)
    p.add_argument('--obs-x', type=float, default=0.30)
    p.add_argument('--obs-y', type=float, default=-1.00)
    p.add_argument('--obs-half-width', type=float, default=0.14)
    p.add_argument('--obs-half-height', type=float, default=0.22)
    p.add_argument('--obstacle-buffer', type=float, default=0.05)
    p.add_argument('--no-show', action='store_true')
    p.add_argument('--save-full-fig', type=str, default=None,
                   help='Optional output path for the final 2-panel figure')
    p.add_argument('--world-legend', choices=['inside', 'outside_left_top', 'none'], default='inside',
                   help='Legend placement for the world/trajectory panel')

    p.add_argument('--verify', action='store_true')
    p.add_argument('--verify-every', type=int, default=1)
    p.add_argument('--verify-uncertainty', type=float, default=0.03)
    p.add_argument('--verify-disturbance', type=float, default=0.03)

    p.add_argument('--repair-on-immrax', action='store_true')
    p.add_argument('--repair-max-iters', type=int, default=6)
    p.add_argument('--repair-speed-backoff', type=float, default=0.08)
    p.add_argument('--repair-buffer-step', type=float, default=0.012)
    p.add_argument('--repair-push-safe', action='store_true',
                   help='After first SAFE repair candidate, try extra conservative backoff while staying SAFE')
    p.add_argument('--repair-push-iters', type=int, default=2)
    p.add_argument('--repair-push-backoff', type=float, default=0.03)
    p.add_argument('--repair-push-k1', action='store_true',
                   help='Also try steering-parameter pushes (k1) during post-safe push')
    p.add_argument('--repair-push-k1-step', type=float, default=0.04)
    p.add_argument('--repair-push-k1-dir', choices=['auto', 'left', 'right'], default='auto')
    return p.parse_args()


def make_rect_polygon(x_lo, x_hi, y_lo, y_hi):
    xs = [x_lo, x_hi, x_hi, x_lo, x_lo]
    ys = [y_lo, y_lo, y_hi, y_hi, y_lo]
    return np.array([xs, ys], dtype=float)


def _truncate_reference(T, U, Z, t_exec):
    t_exec = float(np.clip(t_exec, 0.0, float(T[-1])))
    if t_exec >= float(T[-1]) - 1e-12:
        return T, U, Z
    keep = T < t_exec
    T_new = np.concatenate([T[keep], np.array([t_exec])])
    U_new = np.vstack([np.interp(T_new, T, U[i, :]) for i in range(U.shape[0])])
    Z_new = np.vstack([np.interp(T_new, T, Z[i, :]) for i in range(Z.shape[0])])
    return T_new, U_new, Z_new


def _compute_frs_contour(frs, k_opt, initial_pose, grid_res=100):
    z1g, z2g = np.meshgrid(np.linspace(-1, 1, grid_res), np.linspace(-1, 1, grid_res))
    z_grid = np.vstack([z1g.ravel(), z2g.ravel()])
    pows, coef = frs['pows'], frs['coef']
    z_cols, k_cols = frs['z_cols'], frs['k_cols']
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
    return FRS_to_world(c_frs, initial_pose, frs['initial_x'], frs['initial_y'], frs['distance_scale'])


def _solve_step(frs, fp, state, speed, goal_world, spacing, obs_poly, k_init, obs_buffer):
    z_goal_local = np.asarray(world_to_local(state, goal_world)).reshape(-1)
    x_des_loc, y_des_loc = float(z_goal_local[0]), float(z_goal_local[1])

    o_frs, _, _ = compute_turtlebot_discretized_obs(obs_poly, state, obs_buffer, spacing, frs)
    if o_frs.shape[1] > 0:
        cons_poly = evaluate_frs_polynomial_on_obstacle_points(fp, o_frs)
        cons_grad = get_constraint_polynomial_gradient(cons_poly)
        constraints = [build_constraint(cons_poly, cons_grad)]
    else:
        constraints = []

    v_cur = float(np.clip(speed, frs['v_range'][0], frs['v_range'][1]))
    v_des_lo = max(v_cur - frs['delta_v'], frs['v_range'][0])
    v_des_hi = min(v_cur + frs['delta_v'], frs['v_range'][1])
    v_max = frs['v_range'][1]
    k2_lo = (v_des_lo - v_max / 2.0) * (2.0 / v_max)
    k2_hi = (v_des_hi - v_max / 2.0) * (2.0 / v_max)
    bounds = Bounds(lb=[-1.0, k2_lo], ub=[1.0, k2_hi])

    res = minimize(
        fun=lambda k: turtlebot_cost_and_grad(k, frs['w_max'], v_max, x_des_loc, y_des_loc),
        x0=k_init,
        jac=True,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 220, 'ftol': 1e-6, 'disp': False},
    )
    feasible = bool(res.success or res.status == 0)
    return feasible, (res.x if feasible else None), res


def _rect_separation(a, b):
    dx = max(float(b[0]) - float(a[1]), float(a[0]) - float(b[1]), 0.0)
    dy = max(float(b[2]) - float(a[3]), float(a[2]) - float(b[3]), 0.0)
    return float(np.hypot(dx, dy))


def _tube_clearance(vres, fallback_obs):
    xy = np.asarray(vres.get('xy_tube', np.zeros((0, 4))), dtype=float)
    obs = list(vres.get('expanded_obs', fallback_obs))
    if xy.shape[0] == 0 or len(obs) == 0:
        return float('inf')
    dmin = float('inf')
    for r in xy:
        rr = (float(r[0]), float(r[1]), float(r[2]), float(r[3]))
        for o in obs:
            dmin = min(dmin, _rect_separation(rr, o))
    return dmin


def main():
    args = _parse_args()
    if args.verify_every < 1:
        raise ValueError('--verify-every must be >= 1')

    frs_no = load_frs(path=_FRS_PATHS['noerror'])
    frs_std = load_frs(path=_FRS_PATHS['standard'])
    fp_no = get_frs_polynomial_structure(frs_no['pows'], frs_no['coef'], frs_no['z_cols'], frs_no['k_cols'])
    fp_std = get_frs_polynomial_structure(frs_std['pows'], frs_std['coef'], frs_std['z_cols'], frs_std['k_cols'])

    agent = TurtlebotAgent()
    agent.reset([args.x0, args.y0, args.h0, args.v0])
    goal_world = np.array([[args.x_des], [args.y_des]], dtype=float)

    o = make_rect_polygon(
        args.obs_x - args.obs_half_width, args.obs_x + args.obs_half_width,
        args.obs_y - args.obs_half_height, args.obs_y + args.obs_half_height,
    )
    obstacle_rects = [(
        args.obs_x - args.obs_half_width, args.obs_x + args.obs_half_width,
        args.obs_y - args.obs_half_height, args.obs_y + args.obs_half_height,
    )]

    spacing = compute_turtlebot_point_spacing(agent.footprint, args.obstacle_buffer)
    _, o_buf, o_pts = compute_turtlebot_discretized_obs(o, agent.state[:, -1], args.obstacle_buffer, spacing, frs_no)

    immrax_verify = None
    if args.verify:
        from immrax_verify import verify as immrax_verify

    k_no_prev = np.zeros(2)
    k_std_prev = np.zeros(2)
    no_feas = []
    std_feas = []
    verify_steps_initial, verify_safe_initial = [], []
    verify_traces_initial = []
    verify_steps_final, verify_safe_final = [], []
    verify_traces_final = []
    unsafe_counterfactual_traces = []
    contour_hist = []
    status = 'max_steps_reached'

    print('Starting angled-obstacle compare journey (execute noerror, report standard counterfactual)')

    for step in range(args.max_steps):
        pose = agent.pose
        speed = agent.speed
        dist_goal = float(np.linalg.norm(pose[:2] - goal_world.ravel()))
        print(f'[step {step:02d}] pose=({pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f}), v={speed:.3f}, d_goal={dist_goal:.3f}')

        if dist_goal <= args.goal_tol and speed <= args.speed_tol:
            status = 'goal_reached'
            break

        feas_no, k_no, res_no = _solve_step(
            frs_no, fp_no, agent.state[:, -1], speed, goal_world, spacing, o, k_no_prev, args.obstacle_buffer
        )
        feas_std, k_std, res_std = _solve_step(
            frs_std, fp_std, agent.state[:, -1], speed, goal_world, spacing, o, k_std_prev, args.obstacle_buffer
        )
        no_feas.append(feas_no)
        std_feas.append(feas_std)

        print(f"  noerror  -> {('k=' + str(k_no)) if feas_no else ('infeasible (' + str(res_no.message) + ')')}")
        print(f"  standard -> {('k=' + str(k_std)) if feas_std else ('infeasible (' + str(res_std.message) + ')')}")
        if feas_no and (not feas_std):
            print('  compare: noerror feasible while standard is infeasible')

        if not feas_no:
            status = 'noerror_infeasible'
            break

        w_des, v_des = k_to_wv(k_no, frs_no)
        t_plan = frs_no['t_plan']
        t_stop = v_des / agent.max_accel

        run_verify = args.verify and ((step % args.verify_every == 0) or args.repair_on_immrax)
        if run_verify:
            def _verify_for_k(k_vec):
                vw, vv = k_to_wv(k_vec, frs_no)
                return immrax_verify(
                    w_des=vw,
                    v_des=vv,
                    t_plan=t_plan,
                    t_stop=vv / agent.max_accel,
                    z0=agent.state[:, -1].tolist(),
                    obstacle_rects=obstacle_rects,
                    robot_radius=agent.footprint,
                    obstacle_inflate_radius=0.0,
                    init_uncertainty=args.verify_uncertainty,
                    disturbance_bound=args.verify_disturbance,
                )

            def _post_safe_push(k_base, v_base):
                if not args.repair_push_safe:
                    return np.array(k_base, dtype=float), v_base
                k_push = np.array(k_base, dtype=float)
                v_push = v_base
                c_push = _tube_clearance(v_push, obstacle_rects)
                for _ in range(max(0, args.repair_push_iters)):
                    candidates = []
                    k_c2 = k_push.copy()
                    k_c2[1] = np.clip(k_c2[1] - args.repair_push_backoff, -1.0, 1.0)
                    v_c2 = _verify_for_k(k_c2)
                    if v_c2['safe']:
                        candidates.append((k_c2, v_c2, _tube_clearance(v_c2, obstacle_rects)))
                    if args.repair_push_k1:
                        dirs = [1.0, -1.0]
                        if args.repair_push_k1_dir == 'left':
                            dirs = [1.0]
                        elif args.repair_push_k1_dir == 'right':
                            dirs = [-1.0]
                        for dsgn in dirs:
                            k_c1 = k_push.copy()
                            k_c1[0] = np.clip(k_c1[0] + dsgn * args.repair_push_k1_step, -1.0, 1.0)
                            v_c1 = _verify_for_k(k_c1)
                            if v_c1['safe']:
                                candidates.append((k_c1, v_c1, _tube_clearance(v_c1, obstacle_rects)))
                    if len(candidates) == 0:
                        break
                    k_best, v_best, c_best = max(candidates, key=lambda t: t[2])
                    if c_best <= c_push + 1e-6:
                        break
                    k_push, v_push, c_push = k_best, v_best, c_best
                return k_push, v_push

            vres = _verify_for_k(k_no)
            vres_initial = vres
            verify_steps_initial.append(step)
            verify_safe_initial.append(bool(vres_initial['safe']))
            verify_traces_initial.append({
                'step': int(step),
                'safe': bool(vres_initial['safe']),
                'xy_tube': np.asarray(vres_initial.get('xy_tube', np.zeros((0, 4))), dtype=float),
                'ts_tube': np.asarray(vres_initial.get('ts_tube', np.zeros((0,))), dtype=float),
                'collision_time': (None if vres_initial.get('collision_time') is None
                                   else float(vres_initial.get('collision_time'))),
                'expanded_obs': list(vres_initial.get('expanded_obs', [])),
                'collision_info': vres_initial.get('collision_info', None),
                'nom_xy': np.asarray(vres_initial.get('nom_xy', np.zeros((0, 2))), dtype=float),
            })
            print(f"  immrax(step {step:02d}) -> {'SAFE' if vres['safe'] else 'COLLISION'}")

            if args.repair_on_immrax and (not vres['safe']):
                unsafe_counterfactual_traces.append({
                    'step': int(step),
                    'nom_xy': np.asarray(vres_initial.get('nom_xy', np.zeros((0, 2))), dtype=float),
                    'xy_tube': np.asarray(vres_initial.get('xy_tube', np.zeros((0, 4))), dtype=float),
                    'ts_tube': np.asarray(vres_initial.get('ts_tube', np.zeros((0,))), dtype=float),
                    'collision_time': (None if vres_initial.get('collision_time') is None
                                       else float(vres_initial.get('collision_time'))),
                    'expanded_obs': list(vres_initial.get('expanded_obs', [])),
                    'collision_info': vres_initial.get('collision_info', None),
                })
                repaired = False
                k_cur = np.array(k_no, dtype=float)
                for ridx in range(args.repair_max_iters):
                    k_try = k_cur.copy()
                    k_try[1] = np.clip(k_try[1] - args.repair_speed_backoff, -1.0, 1.0)
                    v_try = _verify_for_k(k_try)
                    print(f"    repair {ridx+1}: speed-backoff k2={k_try[1]:.3f}, "
                          f"immrax={'SAFE' if v_try['safe'] else 'COLLISION'}")
                    if v_try['safe']:
                        k_no = k_try
                        vres = v_try
                        k_no, vres = _post_safe_push(k_no, vres)
                        repaired = True
                        break

                    buf = args.obstacle_buffer + (ridx + 1) * args.repair_buffer_step
                    feas_fix, k_fix, res_fix = _solve_step(
                        frs_no, fp_no, agent.state[:, -1], speed, goal_world, spacing, o, k_try, buf
                    )
                    if not feas_fix:
                        print(f"    repair {ridx+1}: tightened solve infeasible ({res_fix.message})")
                        continue
                    v_fix = _verify_for_k(k_fix)
                    print(f"    repair {ridx+1}: tightened buffer={buf:.3f}, "
                          f"immrax={'SAFE' if v_fix['safe'] else 'COLLISION'}")
                    k_cur = k_fix
                    if v_fix['safe']:
                        k_no = k_fix
                        vres = v_fix
                        k_no, vres = _post_safe_push(k_no, vres)
                        repaired = True
                        break

                if not repaired and not vres['safe']:
                    status = 'immrax_unsafe_after_repair'
                    verify_steps_final.append(step)
                    verify_safe_final.append(False)
                    verify_traces_final.append({
                        'step': int(step),
                        'safe': False,
                        'xy_tube': np.asarray(vres.get('xy_tube', np.zeros((0, 4))), dtype=float),
                        'ts_tube': np.asarray(vres.get('ts_tube', np.zeros((0,))), dtype=float),
                        'collision_time': (None if vres.get('collision_time') is None
                                           else float(vres.get('collision_time'))),
                        'expanded_obs': list(vres.get('expanded_obs', [])),
                        'collision_info': vres.get('collision_info', None),
                        'nom_xy': np.asarray(vres.get('nom_xy', np.zeros((0, 2))), dtype=float),
                    })
                    print('  repair failed: no immrax-safe candidate found; stopping')
                    break

            if status != 'immrax_unsafe_after_repair':
                verify_steps_final.append(step)
                verify_safe_final.append(bool(vres['safe']))
                verify_traces_final.append({
                    'step': int(step),
                    'safe': bool(vres['safe']),
                    'xy_tube': np.asarray(vres.get('xy_tube', np.zeros((0, 4))), dtype=float),
                    'ts_tube': np.asarray(vres.get('ts_tube', np.zeros((0,))), dtype=float),
                    'collision_time': (None if vres.get('collision_time') is None
                                       else float(vres.get('collision_time'))),
                    'expanded_obs': list(vres.get('expanded_obs', [])),
                    'collision_info': vres.get('collision_info', None),
                    'nom_xy': np.asarray(vres.get('nom_xy', np.zeros((0, 2))), dtype=float),
                })

        if status == 'immrax_unsafe_after_repair':
            break

        w_des, v_des = k_to_wv(k_no, frs_no)
        t_stop = v_des / agent.max_accel
        t_ref, u_ref, z_ref = make_turtlebot_braking_trajectory(t_plan, t_stop, w_des, v_des)
        t_exec_ref, u_exec, z_exec = _truncate_reference(t_ref, u_ref, z_ref, min(args.t_move, float(t_ref[-1])))
        n_prev = agent.state.shape[1]
        agent.move(t_exec_ref[-1], t_exec_ref, u_exec, z_exec)

        seg_xy = agent.state[:2, n_prev - 1:]
        seg_dist = np.linalg.norm(seg_xy - goal_world, axis=0)
        hit = np.where(seg_dist <= args.goal_tol)[0]
        if hit.size > 0:
            gidx = (n_prev - 1) + int(hit[0])
            agent.state = agent.state[:, :gidx + 1]
            agent.time = agent.time[:gidx + 1]
            status = 'goal_reached'
            print(f'  goal reached during step at t={agent.time[-1]:.3f}s')
            break

        c_world = _compute_frs_contour(frs_no, k_no, agent.state[:, n_prev - 1], grid_res=90)
        if c_world is not None:
            contour_hist.append(c_world)

        k_no_prev = k_no
        if feas_std:
            k_std_prev = k_std

    _plot(agent, goal_world.ravel(), o, o_buf, o_pts, contour_hist,
          np.array(no_feas, dtype=bool), np.array(std_feas, dtype=bool),
          np.array(verify_steps_initial, dtype=int), np.array(verify_safe_initial, dtype=bool),
          verify_traces_initial,
          np.array(verify_steps_final, dtype=int), np.array(verify_safe_final, dtype=bool),
          verify_traces_final,
          unsafe_counterfactual_traces, status,
          save_full_fig=args.save_full_fig,
          world_legend=args.world_legend)

    if args.no_show:
        plt.close('all')
    else:
        plt.show()


def _plot(agent, goal_world, o, o_buf, o_pts, contour_hist, no_feas, std_feas,
          verify_steps_initial, verify_safe_initial, verify_traces_initial,
          verify_steps_final, verify_safe_final, verify_traces_final,
          unsafe_counterfactual_traces, status, save_full_fig=None, world_legend='inside'):
    def _rect_overlap(a, b):
        return (a[0] <= b[1] and a[1] >= b[0] and a[2] <= b[3] and a[3] >= b[2])

    def _favor_left_subplot(ax_left, ax_right, right_scale, min_gap=0.035):
        pos_left = ax_left.get_position()
        pos_right = ax_right.get_position()
        new_right_width = pos_right.width * right_scale
        new_right_x0 = pos_right.x1 - new_right_width
        new_left_width = max(pos_left.width, new_right_x0 - min_gap - pos_left.x0)
        ax_left.set_position([pos_left.x0, pos_left.y0, new_left_width, pos_left.height])
        ax_right.set_position([new_right_x0, pos_right.y0, new_right_width, pos_right.height])

    obs_rect = (
        float(np.min(o[0])), float(np.max(o[0])),
        float(np.min(o[1])), float(np.max(o[1])),
    )

    fig_width = 18.6 if world_legend == 'outside_left_top' else 16.8
    fig, axes = plt.subplots(
        1, 2, figsize=(fig_width, 6.0),
        gridspec_kw={'width_ratios': [1.20, 0.88]},
    )
    ax = axes[0]
    ax.set_title('Angled Obstacle', fontsize=26, fontweight='medium')
    ax.set_xlabel('x [m]', fontsize=25, fontweight='medium')
    ax.set_ylabel('y [m]', fontsize=25, fontweight='medium')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=24)

    ax.fill(o[0], o[1], color=[0.75, 0.25, 0.25], alpha=0.9, label='_nolegend_')
    ax.fill(o_buf[0], o_buf[1], color=[1.0, 0.55, 0.55], alpha=0.5, label='Buffered Obstacle')
    ax.plot(o_pts[0], o_pts[1], '.', color=[0.4, 0.05, 0.05], markersize=4, label='Obstacle Perimeter Points')

    st = agent.state
    ax.plot(st[0, :], st[1, :], color='purple', linestyle='--', linewidth=2.3, label='Executed RTD Path')
    ax.plot(st[0, 0], st[1, 0], 'ko', markerfacecolor='none', label='_nolegend_')
    ax.plot(st[0, -1], st[1, -1], 'ko', markersize=6, label='_nolegend_')
    ax.plot(goal_world[0], goal_world[1], 'k*', markersize=16, label='_nolegend_')

    max_traces = 7
    if len(verify_traces_final) > max_traces:
        idx = np.unique(np.linspace(0, len(verify_traces_final) - 1, num=max_traces, dtype=int))
        traces = [verify_traces_final[i] for i in idx]
    else:
        traces = verify_traces_final
    for i, tr in enumerate(traces):
        xy_tube = tr['xy_tube']
        nom_xy = tr['nom_xy']
        if xy_tube.shape[0] == 0:
            continue
        base = np.array([0.15, 0.45, 0.95]) if tr['safe'] else np.array([0.90, 0.20, 0.20])
        fade = 0.45 + 0.50 * (i + 1) / max(len(traces), 1)
        col = tuple(np.clip(base * fade, 0.0, 1.0))
        stride = max(1, int(np.ceil(xy_tube.shape[0] / 42.0)))
        for k in range(0, xy_tube.shape[0], stride):
            x0, x1, y0, y1 = [float(v) for v in xy_tube[k]]
            bx = [x0, x1, x1, x0, x0]
            by = [y0, y0, y1, y1, y0]
            ax.plot(bx, by, color=col, linewidth=1.0, alpha=0.28, zorder=5)
        if nom_xy.shape[0] > 0:
            ax.plot(nom_xy[:, 0], nom_xy[:, 1], color=col, linestyle=':', linewidth=1.45, alpha=0.95, zorder=6)
    if traces:
        ax.plot([], [], color=[0.15, 0.45, 0.95], linewidth=1.4, alpha=0.45, label='Immrax Tube Over-Approx. (Safe)')
        ax.plot([], [], color=[0.90, 0.20, 0.20], linewidth=1.4, alpha=0.45, label='Immrax Tube Over-Approx. (Collision)')

    # Counterfactual traces: what would have run without immrax-triggered repair.
    for tr in unsafe_counterfactual_traces:
        nom = tr.get('nom_xy', np.zeros((0, 2)))
        xy_tube = tr.get('xy_tube', np.zeros((0, 4)))
        if xy_tube.shape[0] > 0:
            stride = max(1, int(np.ceil(xy_tube.shape[0] / 42.0)))
            for k in range(0, xy_tube.shape[0], stride):
                x0, x1, y0, y1 = [float(v) for v in xy_tube[k]]
                box = (x0, x1, y0, y1)
                hit = _rect_overlap(box, obs_rect)
                bx = [x0, x1, x1, x0, x0]
                by = [y0, y0, y1, y1, y0]
                ax.plot(
                    bx, by,
                    color='red',
                    linewidth=1.35 if hit else 1.0,
                    alpha=0.9 if hit else 0.22,
                    zorder=8 if hit else 6,
                )
        if nom.shape[0] > 0:
            ax.plot(nom[:, 0], nom[:, 1], color='red', linestyle='--', linewidth=2.2,
                    alpha=0.75, zorder=7)
    if len(unsafe_counterfactual_traces) > 0:
        ax.plot([], [], color='red', linestyle='--', linewidth=2.2,
                label='Unsafe Candidate (Caught by Immrax)')

    if world_legend == 'inside':
        ax.legend(fontsize=16, loc='best')

    ax2 = axes[1]
    ax2.set_title('Per-Step Safety Certification', fontsize=24, fontweight='medium')
    ax2.set_xlabel('Replan Step', fontsize=25, fontweight='medium')
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(['RTD-RAX Final', 'RTD-RAX Initial', 'Standard RTD', 'Non-Inflated RTD'], fontsize=24)
    ax2.grid(True, axis='x', alpha=0.3)
    ax2.tick_params(axis='x', labelsize=24)
    ax2.yaxis.tick_right()
    ax2.tick_params(axis='y', labelleft=False, labelright=True, pad=8)

    steps = np.arange(len(no_feas))
    for s in steps:
        ax2.plot(s, 3, 'o' if no_feas[s] else 'x', color='green' if no_feas[s] else 'red', markersize=8)
        ax2.plot(s, 2, 'o' if std_feas[s] else 'x', color='green' if std_feas[s] else 'orange', markersize=8)
        if no_feas[s] and (not std_feas[s]):
            ax2.axvspan(s - 0.45, s + 0.45, color='gold', alpha=0.25)
    for i, s in enumerate(verify_steps_initial):
        ok = bool(verify_safe_initial[i]) if i < len(verify_safe_initial) else True
        ax2.plot(s, 1, '^' if ok else 'x', color='green' if ok else 'red', markersize=9)
    for i, s in enumerate(verify_steps_final):
        ok = bool(verify_safe_final[i]) if i < len(verify_safe_final) else True
        ax2.plot(s, 0, 's' if ok else 'x', color='green' if ok else 'red', markersize=8)

    ax2.plot([], [], 'o', color='green', label='RTD: Non-Inflated Feasible')
    ax2.plot([], [], 'x', color='orange', label='RTD: Standard Infeasible')
    ax2.plot([], [], '^', color='green', label='Initial RTD-RAX Verdict: SAFE')
    ax2.plot([], [], 's', color='green', label='Final RTD-RAX Verdict: SAFE')
    ax2.plot([], [], 'x', color='red', label='Non-Inflated RTD Infeasible or Immrax Collision')
    ax2.fill_between([], [], [], color='gold', alpha=0.25, label='Non-Inflated Feasible While Standard Is Infeasible')
    if world_legend == 'inside':
        ax2.legend(fontsize=16, loc='best')

    if world_legend == 'outside_left_top':
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        legend_fontsize = 12
        legend_gap = 0.022
        outer_margin = 0.010
        legend_to_axes_gap = 0.070

        # Measure legend widths first, then reserve exactly enough left gutter
        # so both legends stay outside the axes without wasting space.
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92], w_pad=2.8)
        pos1 = ax.get_position()
        probe1 = fig.legend(
            h1, l1, fontsize=legend_fontsize, loc='upper left',
            bbox_to_anchor=(outer_margin, pos1.y1), bbox_transform=fig.transFigure,
            borderaxespad=0.0, frameon=True
        )
        probe2 = fig.legend(
            h2, l2, fontsize=legend_fontsize, loc='upper left',
            bbox_to_anchor=(outer_margin, pos1.y1), bbox_transform=fig.transFigure,
            borderaxespad=0.0, frameon=True
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        probe_bbox1 = probe1.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        probe_bbox2 = probe2.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        max_legend_width = max(probe_bbox1.width, probe_bbox2.width)
        required_left = outer_margin + max_legend_width + legend_to_axes_gap
        probe1.remove()
        probe2.remove()

        fig.tight_layout(rect=[required_left, 0.0, 1.0, 0.92], w_pad=4.8)
        fig.canvas.draw()
        _favor_left_subplot(ax, ax2, right_scale=0.56, min_gap=0.090)
        fig.canvas.draw()
        pos1 = ax.get_position()
        top_legend_y = pos1.y1
        legend_x = max(outer_margin, pos1.x0 - max_legend_width - legend_to_axes_gap)

        leg1 = fig.legend(
            h1, l1, fontsize=legend_fontsize, loc='upper left',
            bbox_to_anchor=(legend_x, top_legend_y), bbox_transform=fig.transFigure,
            borderaxespad=0.0, frameon=True
        )
        fig.canvas.draw()
        bbox1 = leg1.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        lower_legend_y = top_legend_y - bbox1.height - legend_gap
        fig.legend(
            h2, l2, fontsize=legend_fontsize, loc='upper left',
            bbox_to_anchor=(legend_x, lower_legend_y), bbox_transform=fig.transFigure,
            borderaxespad=0.0, frameon=True
        )
    else:
        fig.tight_layout(w_pad=2.8)
        fig.canvas.draw()
        _favor_left_subplot(ax, ax2, right_scale=0.56)
    if save_full_fig:
        fig.savefig(save_full_fig, bbox_inches='tight', pad_inches=0.02)
        print(f'Saved full figure to: {save_full_fig}')


if __name__ == '__main__':
    main()
