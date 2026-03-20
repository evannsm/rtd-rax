"""
rtd_gap_journey_compare.py
==========================
Run a receding-horizon gap journey using the noerror FRS for execution, while
also solving a counterfactual standard-FRS RTD problem at each step.

Goal:
  Demonstrate cases where standard FRS is more conservative (infeasible) while
  noerror FRS remains feasible, and optionally validate executed plans with immrax.
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


OBS_X = 0.75
OBS_HALF_WIDTH = 0.4
GAP_WIDTH = 0.619
OBS_HEIGHT = 0.6
OBSTACLE_BUFFER = 0.05

_FRS_PATHS = {
    'standard': os.path.join(_DEFAULT_DIR, 'turtlebot_FRS_deg_10_v_0_0.5_to_1.0_preproc.mat'),
    'noerror': os.path.join(_DEFAULT_DIR, 'turtlebot_FRS_deg_10_v_0_0.5_to_1.0_noerror_preproc.mat'),
}


def _parse_args():
    p = argparse.ArgumentParser(description='Compare noerror-vs-standard FRS over a full gap journey')
    p.add_argument('--max-steps', type=int, default=60)
    p.add_argument('--goal-tol', type=float, default=0.10)
    p.add_argument('--speed-tol', type=float, default=0.08)
    p.add_argument('--t-move', type=float, default=0.5)
    p.add_argument('--v0', type=float, default=0.75)
    p.add_argument('--x-des', type=float, default=2.0)
    p.add_argument('--y-des', type=float, default=0.0)
    p.add_argument('--no-show', action='store_true')
    p.add_argument('--verify', action='store_true',
                   help='Run immrax checks on the executed noerror plan')
    p.add_argument('--verify-every', type=int, default=1,
                   help='immrax cadence over replans (1 every step, 2 every other, ...)')
    p.add_argument('--verify-uncertainty', type=float, default=0.01)
    p.add_argument('--verify-disturbance', type=float, default=0.0)
    p.add_argument('--repair-on-immrax', action='store_true',
                   help='If immrax flags collision, run repair loop before executing')
    p.add_argument('--repair-max-iters', type=int, default=4,
                   help='Maximum repair iterations per step (default: 4)')
    p.add_argument('--repair-speed-backoff', type=float, default=0.15,
                   help='k2 decrement used by speed-backoff repair (default: 0.15)')
    p.add_argument('--repair-buffer-step', type=float, default=0.01,
                   help='Obstacle buffer increment for CEGIS-style re-solve (default: 0.01 m)')
    p.add_argument('--repair-push-safe', action='store_true',
                   help='After first SAFE repair candidate, try extra conservative backoff while staying SAFE')
    p.add_argument('--repair-push-iters', type=int, default=2,
                   help='Extra post-safe backoff attempts when --repair-push-safe is enabled')
    p.add_argument('--repair-push-backoff', type=float, default=0.03,
                   help='k2 decrement used during post-safe push (default: 0.03)')
    p.add_argument('--repair-push-k1', action='store_true',
                   help='Also try steering-parameter pushes (k1) during post-safe push')
    p.add_argument('--repair-push-k1-step', type=float, default=0.04,
                   help='Absolute k1 step used when --repair-push-k1 is enabled')
    p.add_argument('--repair-push-k1-dir', choices=['auto', 'left', 'right'], default='auto',
                   help='Direction for k1 push; auto tries both signs')
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


def _solve_step(frs, fp, state, speed, goal_world, r, o_upper, o_lower, k_init, obs_buffer):
    z_goal_local = np.asarray(world_to_local(state, goal_world)).reshape(-1)
    x_des_loc, y_des_loc = float(z_goal_local[0]), float(z_goal_local[1])

    parts = []
    for o in (o_upper, o_lower):
        o_frs_i, _, _ = compute_turtlebot_discretized_obs(o, state, obs_buffer, r, frs)
        if o_frs_i.shape[1] > 0:
            parts.append(o_frs_i)
    o_frs = np.hstack(parts) if parts else np.zeros((2, 0))

    if o_frs.shape[1] > 0:
        cons_poly = evaluate_frs_polynomial_on_obstacle_points(fp, o_frs)
        cons_grad = get_constraint_polynomial_gradient(cons_poly)
        constraints_list = [build_constraint(cons_poly, cons_grad)]
    else:
        cons_poly = None
        constraints_list = []

    v_cur = float(np.clip(speed, frs['v_range'][0], frs['v_range'][1]))
    v_des_lo = max(v_cur - frs['delta_v'], frs['v_range'][0])
    v_des_hi = min(v_cur + frs['delta_v'], frs['v_range'][1])
    v_max = frs['v_range'][1]
    k2_lo = (v_des_lo - v_max / 2.0) * (2.0 / v_max)
    k2_hi = (v_des_hi - v_max / 2.0) * (2.0 / v_max)
    bounds = Bounds(lb=[-1.0, k2_lo], ub=[1.0, k2_hi])

    result = minimize(
        fun=lambda k: turtlebot_cost_and_grad(k, frs['w_max'], v_max, x_des_loc, y_des_loc),
        x0=k_init,
        jac=True,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints_list,
        options={'maxiter': 200, 'ftol': 1e-6, 'disp': False},
    )

    feasible = bool(result.success or result.status == 0)
    k_opt = result.x if feasible else None
    return feasible, k_opt, cons_poly, result


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
    if args.repair_max_iters < 1:
        raise ValueError('--repair-max-iters must be >= 1')

    frs_no = load_frs(path=_FRS_PATHS['noerror'])
    frs_std = load_frs(path=_FRS_PATHS['standard'])
    fp_no = get_frs_polynomial_structure(frs_no['pows'], frs_no['coef'], frs_no['z_cols'], frs_no['k_cols'])
    fp_std = get_frs_polynomial_structure(frs_std['pows'], frs_std['coef'], frs_std['z_cols'], frs_std['k_cols'])

    agent = TurtlebotAgent()
    agent.reset([0.0, 0.0, 0.0, args.v0])

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
        o_upper, agent.state[:, -1], OBSTACLE_BUFFER, r, frs_no
    )
    _, o_buf_lower, o_pts_lower = compute_turtlebot_discretized_obs(
        o_lower, agent.state[:, -1], OBSTACLE_BUFFER, r, frs_no
    )

    immrax_verify = None
    if args.verify:
        from immrax_verify import verify as immrax_verify

    k_no_prev = np.zeros(2)
    k_std_prev = np.zeros(2)
    k_no_hist, k_std_hist = [], []
    no_feas, std_feas = [], []
    plan_pose_hist = []
    contour_world_hist = []
    verify_steps_initial, verify_safe_initial, verify_traces_initial = [], [], []
    verify_steps_final, verify_safe_final, verify_traces_final = [], [], []
    unsafe_counterfactual_traces = []
    status = 'max_steps_reached'

    print('Starting compare journey: execute noerror, report standard counterfactual each step')

    for step in range(args.max_steps):
        pose = agent.pose
        speed = agent.speed
        dist_goal = float(np.linalg.norm(pose[:2] - goal_world.ravel()))
        print(f'[step {step:02d}] pose=({pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f}), v={speed:.3f}, d_goal={dist_goal:.3f}')

        if dist_goal <= args.goal_tol and speed <= args.speed_tol:
            status = 'goal_reached'
            break

        feas_no, k_no, cons_no, res_no = _solve_step(
            frs_no, fp_no, agent.state[:, -1], speed, goal_world, r, o_upper, o_lower, k_no_prev,
            obs_buffer=OBSTACLE_BUFFER,
        )
        feas_std, k_std, cons_std, res_std = _solve_step(
            frs_std, fp_std, agent.state[:, -1], speed, goal_world, r, o_upper, o_lower, k_std_prev,
            obs_buffer=OBSTACLE_BUFFER,
        )

        no_feas.append(feas_no)
        std_feas.append(feas_std)
        k_no_hist.append(k_no if feas_no else np.full(2, np.nan))
        k_std_hist.append(k_std if feas_std else np.full(2, np.nan))

        no_txt = f'k={k_no}' if feas_no else f'infeasible ({res_no.message})'
        std_txt = f'k={k_std}' if feas_std else f'infeasible ({res_std.message})'
        print(f'  noerror  -> {no_txt}')
        print(f'  standard -> {std_txt}')

        if feas_no and (not feas_std):
            print('  compare: noerror feasible while standard is infeasible (conservative mismatch)')

        if not feas_no:
            status = 'noerror_infeasible'
            break

        k_no_prev = k_no
        if feas_std:
            k_std_prev = k_std
        plan_pose_hist.append(agent.state[:3, -1].copy())

        w_des, v_des = k_to_wv(k_no, frs_no)
        t_plan = frs_no['t_plan']
        t_stop = v_des / agent.max_accel

        run_verify_this_step = args.verify and ((step % args.verify_every == 0) or args.repair_on_immrax)
        vres = None
        if run_verify_this_step:
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

            # Hybrid repair: speed backoff + CEGIS-style obstacle-buffer tightening.
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
                    # Option A: speed backoff in parameter space.
                    k_try = k_cur.copy()
                    k_try[1] = np.clip(k_try[1] - args.repair_speed_backoff, -1.0, 1.0)
                    v_try = _verify_for_k(k_try)
                    print(f"    repair {ridx+1}: speed-backoff k2 -> {k_try[1]:.3f}, "
                          f"immrax={'SAFE' if v_try['safe'] else 'COLLISION'}")
                    if v_try['safe']:
                        k_no = k_try
                        vres = v_try
                        k_no, vres = _post_safe_push(k_no, vres)
                        repaired = True
                        break

                    # Option B: CEGIS-style tightening by increasing RTD obstacle buffer, then re-solve.
                    buf = OBSTACLE_BUFFER + (ridx + 1) * args.repair_buffer_step
                    feas_fix, k_fix, _, res_fix = _solve_step(
                        frs_no, fp_no, agent.state[:, -1], speed, goal_world, r,
                        o_upper, o_lower, k_try, obs_buffer=buf,
                    )
                    if not feas_fix:
                        print(f"    repair {ridx+1}: tightened-buffer solve infeasible ({res_fix.message})")
                        continue
                    v_fix = _verify_for_k(k_fix)
                    print(f"    repair {ridx+1}: tightened buffer={buf:.3f} -> "
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
                    print('  repair failed: no immrax-safe candidate found; stopping execution')
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
                    break

            # Record final verification result for this step.
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
        hit_idx = np.where(seg_dist <= args.goal_tol)[0]
        if hit_idx.size > 0:
            gidx = (n_prev - 1) + int(hit_idx[0])
            agent.state = agent.state[:, :gidx + 1]
            agent.time = agent.time[:gidx + 1]
            status = 'goal_reached'
            print(f'  goal reached during step at t={agent.time[-1]:.3f}s')
            break

        c_world = _compute_frs_contour(frs_no, k_no, plan_pose_hist[-1], grid_res=90)
        if c_world is not None:
            contour_world_hist.append(c_world)

    k_no_hist = np.array(k_no_hist) if k_no_hist else np.zeros((0, 2))
    k_std_hist = np.array(k_std_hist) if k_std_hist else np.zeros((0, 2))
    no_feas = np.array(no_feas, dtype=bool)
    std_feas = np.array(std_feas, dtype=bool)

    mismatch = np.where(no_feas & (~std_feas))[0]
    print(f'Finished status={status}, mismatch_steps(noerror feasible & standard infeasible)={mismatch.tolist()}')

    _plot_compare(
        agent=agent,
        goal_world=goal_world.ravel(),
        o_upper=o_upper,
        o_lower=o_lower,
        o_buf_upper=o_buf_upper,
        o_buf_lower=o_buf_lower,
        o_pts_upper=o_pts_upper,
        o_pts_lower=o_pts_lower,
        contour_world_hist=contour_world_hist,
        no_feas=no_feas,
        std_feas=std_feas,
        verify_steps_initial=np.array(verify_steps_initial, dtype=int),
        verify_safe_initial=np.array(verify_safe_initial, dtype=bool),
        verify_traces_initial=verify_traces_initial,
        verify_steps_final=np.array(verify_steps_final, dtype=int),
        verify_safe_final=np.array(verify_safe_final, dtype=bool),
        verify_traces_final=verify_traces_final,
        unsafe_counterfactual_traces=unsafe_counterfactual_traces,
        status=status,
    )
    plt.tight_layout()
    if args.no_show:
        plt.close('all')
    else:
        plt.show()


def _plot_compare(agent, goal_world, o_upper, o_lower, o_buf_upper, o_buf_lower, o_pts_upper, o_pts_lower,
                  contour_world_hist, no_feas, std_feas,
                  verify_steps_initial, verify_safe_initial, verify_traces_initial,
                  verify_steps_final, verify_safe_final, verify_traces_final,
                  unsafe_counterfactual_traces, status):
    def _rect_overlap(a, b):
        return (a[0] <= b[1] and a[1] >= b[0] and a[2] <= b[3] and a[3] >= b[2])

    obs_rects = [
        (float(np.min(o_upper[0])), float(np.max(o_upper[0])),
         float(np.min(o_upper[1])), float(np.max(o_upper[1]))),
        (float(np.min(o_lower[0])), float(np.max(o_lower[0])),
         float(np.min(o_lower[1])), float(np.max(o_lower[1]))),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    ax = axes[0]
    ax.set_title(f'Journey Compare (execute noerror) | status: {status}')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    obs_color = [0.75, 0.25, 0.25]
    buf_color = [1.0, 0.55, 0.55]
    pts_color = [0.4, 0.05, 0.05]
    for o_raw, o_buf, o_pts in ((o_upper, o_buf_upper, o_pts_upper), (o_lower, o_buf_lower, o_pts_lower)):
        ax.fill(o_raw[0], o_raw[1], color=obs_color, alpha=0.9, label='_nolegend_')
        ax.fill(o_buf[0], o_buf[1], color=buf_color, alpha=0.5, label='_nolegend_')
        ax.plot(o_pts[0], o_pts[1], '.', color=pts_color, markersize=3, label='_nolegend_')
    ax.fill([], [], color=obs_color, alpha=0.9, label='obstacle')
    ax.fill([], [], color=buf_color, alpha=0.5, label='buffered obstacle')
    ax.plot([], [], '.', color=pts_color, markersize=4, label='obs pts')

    for c_world in contour_world_hist:
        ax.plot(c_world[0], c_world[1], color=[0.25, 0.8, 0.35], alpha=0.18, linewidth=1.0)
    if contour_world_hist:
        ax.plot([], [], color=[0.25, 0.8, 0.35], alpha=0.7, linewidth=1.2, label='noerror FRS @ replans')

    st = agent.state
    ax.plot(st[0, :], st[1, :], color='purple', linestyle='--', linewidth=2.0, label='executed RTD path')
    ax.plot(st[0, 0], st[1, 0], 'ko', markerfacecolor='none', label='start')
    ax.plot(st[0, -1], st[1, -1], 'ko', markersize=5, label='final')
    ax.plot(goal_world[0], goal_world[1], 'k*', markersize=14, label='goal')

    # Immrax overlay (box-based for both SAFE and COLLISION to avoid projection mismatch)
    max_traces = 7
    traces = verify_traces_final if len(verify_traces_final) <= max_traces else [verify_traces_final[i] for i in np.unique(
        np.linspace(0, len(verify_traces_final) - 1, num=max_traces, dtype=int)
    )]
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
            ax.plot(bx, by, color=col, linewidth=0.9, alpha=0.22, zorder=5)
        if nom_xy.shape[0] > 0:
            ax.plot(nom_xy[:, 0], nom_xy[:, 1], color=col, linestyle=':', linewidth=1.3, alpha=0.95, zorder=6)
    if traces:
        ax.plot([], [], color=[0.15, 0.45, 0.95], linewidth=1.2, alpha=0.35, label='Immrax SAFE tube boxes')
        ax.plot([], [], color=[0.90, 0.20, 0.20], linewidth=1.2, alpha=0.35, label='Immrax COLLISION tube boxes')

    for tr in unsafe_counterfactual_traces:
        nom = tr.get('nom_xy', np.zeros((0, 2)))
        xy_tube = tr.get('xy_tube', np.zeros((0, 4)))
        if xy_tube.shape[0] > 0:
            stride = max(1, int(np.ceil(xy_tube.shape[0] / 42.0)))
            for k in range(0, xy_tube.shape[0], stride):
                x0, x1, y0, y1 = [float(v) for v in xy_tube[k]]
                box = (x0, x1, y0, y1)
                hit = any(_rect_overlap(box, obs) for obs in obs_rects)
                bx = [x0, x1, x1, x0, x0]
                by = [y0, y0, y1, y1, y0]
                ax.plot(
                    bx, by,
                    color='red',
                    linewidth=1.2 if hit else 0.9,
                    alpha=0.9 if hit else 0.22,
                    zorder=8 if hit else 6,
                )
        if nom.shape[0] > 0:
            ax.plot(nom[:, 0], nom[:, 1], color='red', linestyle='--', linewidth=2.0,
                    alpha=0.75, zorder=7)
    if len(unsafe_counterfactual_traces) > 0:
        ax.plot([], [], color='red', linestyle='--', linewidth=2.0,
                label='unsafe candidate (caught by Immrax)')
        ax.plot([], [], color='red', linewidth=1.2, alpha=0.9, label='unsafe tube boxes (colliding highlighted)')

    ax.legend(fontsize=8, loc='best')

    # Step-wise comparison timeline
    ax2 = axes[1]
    ax2.set_title('Per-Step Feasibility Comparison')
    ax2.set_xlabel('replan step')
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(['Immrax final', 'Immrax initial', 'standard FRS', 'noerror FRS'])
    ax2.grid(True, axis='x', alpha=0.3)

    n = len(no_feas)
    steps = np.arange(n)
    for s in steps:
        ax2.plot(s, 3, 'o' if no_feas[s] else 'x', color='green' if no_feas[s] else 'red', markersize=7)
        ax2.plot(s, 2, 'o' if std_feas[s] else 'x', color='green' if std_feas[s] else 'orange', markersize=7)
        if no_feas[s] and (not std_feas[s]):
            ax2.axvspan(s - 0.45, s + 0.45, color='gold', alpha=0.25)

    for i, s in enumerate(verify_steps_initial):
        safe = bool(verify_safe_initial[i]) if i < len(verify_safe_initial) else True
        ax2.plot(s, 1, '^' if safe else 'x', color='green' if safe else 'red', markersize=8)
    for i, s in enumerate(verify_steps_final):
        safe = bool(verify_safe_final[i]) if i < len(verify_safe_final) else True
        ax2.plot(s, 0, 's' if safe else 'x', color='green' if safe else 'red', markersize=7)

    ax2.plot([], [], 'o', color='green', label='feasible / SAFE')
    ax2.plot([], [], 'x', color='orange', label='standard infeasible')
    ax2.plot([], [], '^', color='green', label='Immrax initial SAFE')
    ax2.plot([], [], 's', color='green', label='Immrax final SAFE')
    ax2.plot([], [], 'x', color='red', label='noerror infeasible / Immrax COLLISION')
    ax2.fill_between([], [], [], color='gold', alpha=0.25,
                     label='noerror feasible & standard infeasible')
    ax2.legend(fontsize=8, loc='best')


if __name__ == '__main__':
    main()
