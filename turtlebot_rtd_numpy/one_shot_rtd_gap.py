"""
one_shot_rtd_gap.py  –  NumPy / SciPy version
================================================
One-shot RTD demonstration with a two-obstacle gap scenario.

Two rectangular obstacles are placed symmetrically about y=0 at a fixed
x location, leaving a gap that only the no-error FRS can fit through.
Change FRS_VERSION to compare the standard (tracking-error-aware) FRS
against the no-error FRS.

At x=0.75 m (k=[0,0]):
  standard FRS half-width ≈ 0.152 m  → needs gap ≥ 0.654 m to pass
  no-error FRS half-width ≈ 0.094 m  → needs gap ≥ 0.538 m to pass
  GAP_WIDTH = 0.619 m sits between these, so noerror passes, standard does not.
"""

import sys
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
from matplotlib.lines import Line2D
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
    'legend.fontsize': 22,
    'figure.titlesize': 27,
    'figure.titleweight': 'medium',
})

sys.path.insert(0, os.path.dirname(__file__))

from frs_loader       import load_frs, k_to_wv, _DEFAULT_DIR
from geometry_utils   import (world_to_local, FRS_to_world,
                               compute_turtlebot_point_spacing,
                               compute_turtlebot_discretized_obs)
from trajectory       import make_turtlebot_braking_trajectory
from turtlebot_agent  import TurtlebotAgent
from polynomial_utils import (get_frs_polynomial_structure,
                               evaluate_frs_polynomial_on_obstacle_points,
                               get_constraint_polynomial_gradient,
                               eval_constraint_poly)
from cost             import turtlebot_cost_and_grad
from constraints      import build_constraint


# ===========================================================================
# User parameters
# ===========================================================================

def _parse_args():
    p = argparse.ArgumentParser(description='Gap scenario RTD demo')
    p.add_argument('--frs', choices=['standard', 'noerror'], default='standard',
                   help='FRS version to use (default: standard)')
    p.add_argument('--verify', action='store_true',
                   help='Run immrax reachability verification after planning')
    p.add_argument('--uncertainty', type=float, default=0.01,
                   help='Additional positional uncertainty for immrax (default: 0.01 m). '
                        'Total position uncertainty = footprint + this value; heading/speed use 1/5 of total')
    p.add_argument('--disturbance', type=float, default=0.0,
                   help='Bounded additive disturbance on each state derivative for immrax '
                        '(default: 0.0)')
    p.add_argument('--save-world-fig', type=str, default=None,
                   help='Optional output path for the standalone world-frame PDF/figure')
    p.add_argument('--save-full-fig', type=str, default=None,
                   help='Optional output path for the full 3-panel figure PDF/figure')
    p.add_argument('--compact-full-fig', action='store_true',
                   help='Use a smaller Case 1 full-figure layout with only k-space and world-frame panels')
    p.add_argument('--compact-full-layout', choices=['vertical', 'horizontal'],
                   default='vertical',
                   help='Layout for the compact Case 1 full-figure export')
    p.add_argument('--compact-full-legend-placement', choices=['bottom', 'right', 'left'],
                   default='bottom',
                   help='Legend placement for compact full figures when a legend is shown')
    p.add_argument('--world-legend', choices=['inside', 'outside', 'none'], default='inside',
                   help='Legend placement for the standalone world-frame figure')
    p.add_argument('--panel-world-legend', choices=['inside', 'outside_top', 'none'], default='inside',
                   help='Legend placement for the world panel inside the 3-panel figure')
    p.add_argument('--minimal-world-text', action='store_true',
                   help='Only keep the PATH FOUND / NO PATH FOUND text inside the world-frame axes')
    p.add_argument('--hide-result-text', action='store_true',
                   help='Hide the PATH FOUND / NO PATH FOUND text inside the world-frame axes')
    p.add_argument('--hide-verify-text', action='store_true',
                   help='Hide the immrax verdict text inside the world-frame axes')
    p.add_argument('--custom-title', type=str, default=None,
                   help='Optional custom figure title override')
    p.add_argument('--world-title', type=str, default='',
                   help='Optional title for the standalone world-frame figure')
    p.add_argument('--font-bump', type=int, default=0,
                   help='Per-export font size adjustment in points')
    p.add_argument('--world-label-bump', type=int, default=0,
                   help='Additional font size adjustment for world-frame axis labels')
    p.add_argument('--world-tick-bump', type=int, default=0,
                   help='Additional font size adjustment for world-frame tick labels')
    p.add_argument('--suptitle-bump', type=int, default=0,
                   help='Additional font size adjustment for the figure suptitle only')
    p.add_argument('--hide-v0-in-title', action='store_true',
                   help='Remove the V_0 term from the figure title')
    p.add_argument('--hide-gap-annotation', action='store_true',
                   help='Hide the gap dimension annotation and dashed lines')
    p.add_argument('--overlay-noerror-frs', action='store_true',
                   help='Overlay the noerror FRS contour (solved independently) on the plot')
    p.add_argument('--hide-frs-contour', action='store_true',
                   help='Hide the FRS @ k_opt contour from the plot')
    p.add_argument('--hide-immrax-nominal', action='store_true',
                   help='Hide the immrax nominal reference trajectory from the plot')
    p.add_argument('--show-start-footprint', action='store_true',
                   help='Draw the vehicle footprint icon at the start pose')
    p.add_argument('--hide-current-footprint', action='store_true',
                   help='Hide the vehicle footprint icon at the current/final pose')
    p.add_argument('--hide-start-marker', action='store_true',
                   help='Hide the start marker in the world-frame plot')
    p.add_argument('--hide-current-footprint-arrow', action='store_true',
                   help='Hide the arrow on the current/final footprint icon')
    p.add_argument('--footprint-color-mode', choices=['default', 'compare'], default='default',
                   help='Vehicle footprint coloring mode for world-frame exports')
    p.add_argument('--show-goal', action='store_true',
                   help='Extend viewport to include the goal position')
    p.add_argument('--legend-include-nominal-frs', action='store_true',
                   help='Add a Nominal FRS entry to the legend even if contour is hidden')
    p.add_argument('--legend-include-mmr-frs', action='store_true',
                   help='Add an MMR FRS entry to the legend even if contour is hidden')
    p.add_argument('--no-show', action='store_true',
                   help='Skip interactive display and only save figures if requested')
    return p.parse_args()

V_0             = 0.75   # initial speed (m/s) — must be in [0.25, 1.25]

# Goal: straight ahead through the gap
X_DES           = 2.0
Y_DES           = 0.0

# Obstacle geometry
OBS_X           = 0.75   # x-centre of both rectangles (m)
OBS_HALF_WIDTH  = 0.4    # half-width of each rectangle in x (m)
GAP_WIDTH       = 0.619  # clear gap between the two inner edges (m)
OBS_HEIGHT      = 0.6    # height of each rectangle (m), above/below the gap

OBSTACLE_BUFFER = 0.05   # buffer added to obstacle before discretisation (m)


# ===========================================================================
# FRS file paths
# ===========================================================================

_FRS_PATHS = {
    'standard': os.path.join(
        _DEFAULT_DIR,
        'turtlebot_FRS_deg_10_v_0_0.5_to_1.0_preproc.mat'
    ),
    'noerror': os.path.join(
        _DEFAULT_DIR,
        'turtlebot_FRS_deg_10_v_0_0.5_to_1.0_noerror_preproc.mat'
    ),
}


# ===========================================================================
# Obstacle helpers
# ===========================================================================

def make_rect_polygon(x_lo, x_hi, y_lo, y_hi):
    """Return a (2, 5) closed rectangle polygon (CCW)."""
    xs = [x_lo, x_hi, x_hi, x_lo, x_lo]
    ys = [y_lo, y_lo, y_hi, y_hi, y_lo]
    return np.array([xs, ys], dtype=float)


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = _parse_args()
    FRS_VERSION = args.frs
    DO_VERIFY   = args.verify
    UNCERTAINTY = args.uncertainty
    DISTURBANCE = args.disturbance

    # --- Load FRS ---
    frs_path = _FRS_PATHS[FRS_VERSION]
    print(f'Loading FRS ({FRS_VERSION!r})...')
    frs = load_frs(path=frs_path)

    # --- Create agent ---
    agent = TurtlebotAgent()
    agent.reset([0.0, 0.0, 0.0, V_0])

    # --- Two rectangular obstacles ---
    x_lo = OBS_X - OBS_HALF_WIDTH
    x_hi = OBS_X + OBS_HALF_WIDTH
    half_gap = GAP_WIDTH / 2.0

    O_upper = make_rect_polygon(x_lo, x_hi, half_gap,              half_gap + OBS_HEIGHT)
    O_lower = make_rect_polygon(x_lo, x_hi, -half_gap - OBS_HEIGHT, -half_gap)

    print(f'Upper obstacle: y ∈ [{half_gap:.3f}, {half_gap + OBS_HEIGHT:.3f}] m')
    print(f'Lower obstacle: y ∈ [{-half_gap - OBS_HEIGHT:.3f}, {-half_gap:.3f}] m')
    print(f'Gap: {GAP_WIDTH:.3f} m')

    # Raw obstacle bounding boxes in world frame (used by immrax)
    obs_x_lo = OBS_X - OBS_HALF_WIDTH
    obs_x_hi = OBS_X + OBS_HALF_WIDTH
    obstacle_rects = [
        (obs_x_lo, obs_x_hi,  half_gap,              half_gap + OBS_HEIGHT),
        (obs_x_lo, obs_x_hi, -half_gap - OBS_HEIGHT, -half_gap),
    ]

    # --- Goal in robot-local frame (agent starts at origin, heading 0) ---
    z_goal       = np.array([[X_DES], [Y_DES]])
    z_goal_local = np.asarray(world_to_local(agent.state[:, -1], z_goal)).reshape(-1)
    x_des_loc    = float(z_goal_local[0])
    y_des_loc    = float(z_goal_local[1])

    # --- Discretise both obstacles into FRS frame, then combine ---
    r = compute_turtlebot_point_spacing(agent.footprint, OBSTACLE_BUFFER)

    results = []
    for O in (O_upper, O_lower):
        O_FRS_i, O_buf_i, O_pts_i = compute_turtlebot_discretized_obs(
            O, agent.state[:, -1], OBSTACLE_BUFFER, r, frs
        )
        results.append((O_FRS_i, O_buf_i, O_pts_i))

    (O_FRS_upper, O_buf_upper, O_pts_upper) = results[0]
    (O_FRS_lower, O_buf_lower, O_pts_lower) = results[1]

    # Combine FRS-frame obstacle points from both obstacles
    parts = [p for p in (O_FRS_upper, O_FRS_lower) if p.shape[1] > 0]
    O_FRS = np.hstack(parts) if parts else np.zeros((2, 0))

    # --- Build polynomial structures ---
    fp = get_frs_polynomial_structure(
        frs['pows'], frs['coef'], frs['z_cols'], frs['k_cols']
    )

    if O_FRS.shape[1] > 0:
        print(f'Building constraints over {O_FRS.shape[1]} obstacle points...')
        cons_poly  = evaluate_frs_polynomial_on_obstacle_points(fp, O_FRS)
        cons_grad  = get_constraint_polynomial_gradient(cons_poly)
        constraint = build_constraint(cons_poly, cons_grad)
        constraints_list = [constraint]
    else:
        print('No obstacle points inside FRS region.')
        cons_poly = cons_grad = None
        constraints_list = []

    # --- Parameter bounds ---
    w_max    = frs['w_max']
    v_max    = frs['v_range'][1]
    v_des_lo = max(V_0 - frs['delta_v'], frs['v_range'][0])
    v_des_hi = min(V_0 + frs['delta_v'], frs['v_range'][1])
    k_2_lo   = (v_des_lo - v_max / 2.0) * (2.0 / v_max)
    k_2_hi   = (v_des_hi - v_max / 2.0) * (2.0 / v_max)
    bounds   = Bounds(lb=[-1.0, k_2_lo], ub=[1.0, k_2_hi])

    # --- Optimise ---
    print('Running trajectory optimisation...')
    result = minimize(
        fun=lambda k: turtlebot_cost_and_grad(k, w_max, v_max, x_des_loc, y_des_loc),
        x0=np.zeros(2),
        jac=True,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints_list,
        options={'maxiter': int(1e5), 'ftol': 1e-6, 'disp': False},
    )

    k_opt = None
    if result.success or result.status == 0:
        k_opt = result.x
        print(f'Feasible trajectory found!  k_opt = {k_opt}')
    else:
        print(f'No feasible trajectory found.  ({result.message})')

    # --- Braking trajectory ---
    T_brk = U_brk = Z_brk = None
    if k_opt is not None:
        w_des, v_des = k_to_wv(k_opt, frs)
        t_plan = frs['t_plan']
        t_stop = v_des / agent.max_accel
        T_brk, U_brk, Z_brk = make_turtlebot_braking_trajectory(
            t_plan, t_stop, w_des, v_des
        )
        agent.move(T_brk[-1], T_brk, U_brk, Z_brk)
        print(f'Agent moved.  Final pose: {agent.pose}')

    # --- FRS contour at k_opt ---
    C_FRS = C_world = None
    if k_opt is not None:
        C_FRS, C_world = _compute_frs_contour(frs, k_opt, agent.state[:, 0], grid_res=220)
        _print_planning_diagnostics(
            k_opt,
            cons_poly,
            C_world,
            O_pts_upper,
            O_pts_lower,
        )

    # --- immrax verification (optional) ---
    verify_result = None
    if DO_VERIFY:
        from immrax_verify import verify as immrax_verify
        print('\nRunning immrax reachability verification...')

        if k_opt is not None:
            # RTD found a path: verify the planned trajectory is actually safe
            print('  Verifying RTD-planned trajectory (expect SAFE for noerror)...')
            vw_des, vv_des = k_to_wv(k_opt, frs)
            vt_plan = frs['t_plan']
            vt_stop = vv_des / agent.max_accel
        else:
            # RTD found no path: verify a straight-ahead trajectory to show collision
            print('  No RTD path; verifying straight-ahead trajectory (expect COLLISION)...')
            vw_des  = 0.0
            vv_des  = V_0
            vt_plan = frs['t_plan']
            vt_stop = V_0 / agent.max_accel
        print(f'  Horizons: t_plan={vt_plan:.3f} s, t_stop={vt_stop:.3f} s, tf={vt_plan + vt_stop:.3f} s')

        verify_result = immrax_verify(
            w_des          = vw_des,
            v_des          = vv_des,
            t_plan         = vt_plan,
            t_stop         = vt_stop,
            z0             = [0.0, 0.0, 0.0, V_0],
            obstacle_rects = obstacle_rects,
            robot_radius   = agent.footprint,
            obstacle_inflate_radius = 0.0,
            init_uncertainty = UNCERTAINTY,
            disturbance_bound = DISTURBANCE,
        )
        if verify_result is not None and 'uncertainty_vec' in verify_result:
            eps = np.asarray(verify_result['uncertainty_vec']).reshape(-1)
            eps_add = float(verify_result.get('positional_uncertainty_added', UNCERTAINTY))
            eps_tot = float(verify_result.get('positional_uncertainty_total', eps[0]))
            d_bound_raw = verify_result.get('disturbance_bound', DISTURBANCE)
            d_bound = float(np.max(d_bound_raw))
            obs_inf = float(verify_result.get('obstacle_inflate_radius', 0.0))
            print(f'  Positional uncertainty: footprint + added = {agent.footprint:.3f} + {eps_add:.3f} = {eps_tot:.3f}')
            print(f'  Applied initial uncertainty [x, y, heading, v] = {eps}')
            print(f'  Applied disturbance bound [dpx, dpy, dh, dv] = ±{d_bound:.3f}')
            print(f'  Obstacle inflation in collision check = ±{obs_inf:.3f}')
            _print_immrax_tube_growth_diagnostics(verify_result)

    # --- Overlay noerror FRS contour (optional) ---
    overlay_C_world = None
    overlay_mmr_C_world = None
    if args.overlay_noerror_frs and FRS_VERSION != 'noerror':
        print('\nComputing overlay noerror FRS contour...')
        noerror_frs = load_frs(path=_FRS_PATHS['noerror'])
        ne_fp = get_frs_polynomial_structure(
            noerror_frs['pows'], noerror_frs['coef'],
            noerror_frs['z_cols'], noerror_frs['k_cols'],
        )
        ne_parts = []
        for O in (O_upper, O_lower):
            ne_O_FRS_i, _, _ = compute_turtlebot_discretized_obs(
                O, agent.state[:, 0], OBSTACLE_BUFFER, r, noerror_frs,
            )
            if ne_O_FRS_i.shape[1] > 0:
                ne_parts.append(ne_O_FRS_i)
        ne_O_FRS = np.hstack(ne_parts) if ne_parts else np.zeros((2, 0))
        if ne_O_FRS.shape[1] > 0:
            ne_cons_poly = evaluate_frs_polynomial_on_obstacle_points(ne_fp, ne_O_FRS)
            ne_cons_grad = get_constraint_polynomial_gradient(ne_cons_poly)
            ne_constraint = build_constraint(ne_cons_poly, ne_cons_grad)
            ne_constraints_list = [ne_constraint]
        else:
            ne_constraints_list = []
        ne_w_max = noerror_frs['w_max']
        ne_v_max = noerror_frs['v_range'][1]
        ne_v_lo = max(V_0 - noerror_frs['delta_v'], noerror_frs['v_range'][0])
        ne_v_hi = min(V_0 + noerror_frs['delta_v'], noerror_frs['v_range'][1])
        ne_k2_lo = (ne_v_lo - ne_v_max / 2.0) * (2.0 / ne_v_max)
        ne_k2_hi = (ne_v_hi - ne_v_max / 2.0) * (2.0 / ne_v_max)
        ne_bounds = Bounds(lb=[-1.0, ne_k2_lo], ub=[1.0, ne_k2_hi])
        ne_result = minimize(
            fun=lambda k: turtlebot_cost_and_grad(k, ne_w_max, ne_v_max, x_des_loc, y_des_loc),
            x0=np.zeros(2), jac=True, method='SLSQP',
            bounds=ne_bounds, constraints=ne_constraints_list,
            options={'maxiter': int(1e5), 'ftol': 1e-6, 'disp': False},
        )
        if ne_result.success or ne_result.status == 0:
            ne_k_opt = ne_result.x
            _, overlay_C_world = _compute_frs_contour(noerror_frs, ne_k_opt, agent.state[:, 0], grid_res=220)
            print(f'  Noerror FRS overlay: k_opt = {ne_k_opt}, contour computed')
        else:
            print('  Noerror FRS overlay: no feasible solution found')

    # Hide the primary FRS contour if requested
    plot_C_FRS = None if args.hide_frs_contour else C_FRS
    plot_C_world = None if args.hide_frs_contour else C_world

    # --- Plot ---
    _plot(
        agent,
        O_upper, O_lower,
        O_buf_upper, O_buf_lower,
        O_pts_upper, O_pts_lower,
        O_FRS, frs, k_opt,
        T_brk, Z_brk, plot_C_FRS, plot_C_world, cons_poly,
        feasible=(k_opt is not None),
        frs_version=FRS_VERSION,
        verify_result=verify_result,
        save_world_fig=args.save_world_fig,
        save_full_fig=args.save_full_fig,
        world_legend=args.world_legend,
        panel_world_legend=args.panel_world_legend,
        minimal_world_text=args.minimal_world_text,
        hide_result_text=args.hide_result_text,
        hide_verify_text=args.hide_verify_text,
        custom_title=args.custom_title,
        world_title=args.world_title,
        font_bump=args.font_bump,
        world_label_bump=args.world_label_bump,
        world_tick_bump=args.world_tick_bump,
        hide_v0_in_title=args.hide_v0_in_title,
        hide_gap_annotation=args.hide_gap_annotation,
        suptitle_bump=args.suptitle_bump,
        overlay_C_world=overlay_C_world,
        overlay_mmr_C_world=overlay_mmr_C_world,
        show_goal=args.show_goal,
        hide_immrax_nominal=args.hide_immrax_nominal,
        show_start_footprint=args.show_start_footprint,
        hide_current_footprint=args.hide_current_footprint,
        hide_start_marker=args.hide_start_marker,
        hide_current_footprint_arrow=args.hide_current_footprint_arrow,
        footprint_color_mode=args.footprint_color_mode,
        legend_include_nominal_frs=args.legend_include_nominal_frs,
        legend_include_mmr_frs=args.legend_include_mmr_frs,
        compact_full_fig=args.compact_full_fig,
        compact_full_layout=args.compact_full_layout,
        compact_full_legend_placement=args.compact_full_legend_placement,
    )
    plt.tight_layout()
    if not args.no_show:
        plt.show()


# ===========================================================================
# Helpers
# ===========================================================================

def _compute_frs_contour(frs, k_opt, initial_pose, grid_res=100):
    """Evaluate the FRS polynomial at k_opt on a z-grid; extract the contour."""
    z1g, z2g = np.meshgrid(np.linspace(-1, 1, grid_res),
                            np.linspace(-1, 1, grid_res))
    z_grid = np.vstack([z1g.ravel(), z2g.ravel()])

    pows   = frs['pows']
    coef   = frs['coef']
    z_cols = frs['z_cols']
    k_cols = frs['k_cols']

    k_pows  = pows[:, k_cols]
    z_pows  = pows[:, z_cols]
    k_mono  = np.prod(k_opt[np.newaxis, :] ** k_pows, axis=1)
    z_vals  = np.prod(z_grid[np.newaxis, :, :] ** z_pows[:, :, np.newaxis], axis=1)
    frs_grid = (coef * k_mono) @ z_vals

    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(z1g, z2g, frs_grid.reshape(z1g.shape), levels=[0.0])
    plt.close(fig_tmp)

    C_FRS = C_world = None
    if cs.allsegs and cs.allsegs[0]:
        segs_all  = cs.allsegs[0]
        close_tol = 5.0 / float(grid_res)
        segs = [s for s in segs_all
                if s.size > 0 and np.linalg.norm(s[0] - s[-1]) <= close_tol]
        if not segs:
            segs = segs_all

        parts = []
        for i, seg in enumerate(segs):
            parts.append(seg.T)
            if i < len(segs) - 1:
                parts.append(np.full((2, 1), np.nan))
        C_FRS = np.hstack(parts) if parts else None

    if C_FRS is not None:
        x0, y0  = frs['initial_x'], frs['initial_y']
        D       = frs['distance_scale']
        C_world = FRS_to_world(C_FRS, initial_pose, x0, y0, D)

    return C_FRS, C_world


def _print_planning_diagnostics(k_opt, cons_poly, C_world, O_pts_upper, O_pts_lower):
    """Print quantitative safety diagnostics for the chosen k_opt."""
    if cons_poly is not None:
        g_vals = np.asarray(eval_constraint_poly(cons_poly, np.asarray(k_opt))).reshape(-1)
        if g_vals.size > 0:
            max_g = float(np.max(g_vals))
            min_slack = float(-max_g)
            print(f'Diagnostic: max constraint g(k_opt) = {max_g:.3e} (<= 0 is feasible)')
            print(f'Diagnostic: minimum constraint slack = {min_slack:.3e}')

    if C_world is None:
        return

    contour_pts = C_world[:, ~np.isnan(C_world).any(axis=0)]
    obs_parts = [p for p in (O_pts_upper, O_pts_lower) if p is not None and p.shape[1] > 0]
    if contour_pts.shape[1] == 0 or not obs_parts:
        return

    obs_pts = np.hstack(obs_parts)
    delta = contour_pts[:, :, np.newaxis] - obs_pts[:, np.newaxis, :]
    dists = np.linalg.norm(delta, axis=0)
    dmin = float(np.min(dists))
    print(f'Diagnostic: min distance(FRS@k_opt contour, buffered obstacle points) = {dmin:.4f} m')


def _print_immrax_tube_growth_diagnostics(verify_result):
    """Print width/area diagnostics over time for the immrax XY tube."""
    xy_tube = np.asarray(verify_result.get('xy_tube', np.zeros((0, 4))))
    ts_tube = np.asarray(verify_result.get('ts_tube', np.zeros((0,))), dtype=float)
    if xy_tube.shape[0] < 2:
        return

    wx = xy_tube[:, 1] - xy_tube[:, 0]
    wy = xy_tube[:, 3] - xy_tube[:, 2]
    area = wx * wy

    def _fmt_delta(arr):
        return f'{arr[-1] - arr[0]:+.4e}'

    t0 = float(ts_tube[0]) if ts_tube.size else 0.0
    tf = float(ts_tube[-1]) if ts_tube.size else float(xy_tube.shape[0] - 1)
    print(f'  Tube growth diagnostics over t=[{t0:.3f}, {tf:.3f}] s:')
    print(f'    width_x  start={wx[0]:.4e}, end={wx[-1]:.4e}, delta={_fmt_delta(wx)}')
    print(f'    width_y  start={wy[0]:.4e}, end={wy[-1]:.4e}, delta={_fmt_delta(wy)}')
    print(f'    area_xy  start={area[0]:.4e}, end={area[-1]:.4e}, delta={_fmt_delta(area)}')


def _draw_vehicle_icon(ax, x, y, heading, radius, color='steelblue', alpha=0.5,
                       show_center_ring=False, show_arrow=True,
                       edgecolor=None, linewidth=1.0):
    """Draw a Turtlebot footprint icon with optional inner ring."""
    if edgecolor is None:
        edgecolor = color

    circle = mpatches.Circle(
        (x, y), radius,
        facecolor=color, edgecolor=edgecolor,
        linewidth=linewidth, alpha=alpha, zorder=6,
    )
    ax.add_patch(circle)

    if show_center_ring:
        ax.plot(x, y, marker='o', markerfacecolor='none',
                markeredgecolor='black', markersize=9,
                linestyle='None', zorder=7)

    if show_arrow:
        ax.annotate(
            '',
            xy=(x + radius * np.cos(heading), y + radius * np.sin(heading)),
            xytext=(x, y),
            arrowprops=dict(arrowstyle='->', color='k', lw=1.5),
            zorder=7,
        )


def _legend_handles_by_label(ax):
    handles, labels = ax.get_legend_handles_labels()
    by_label = {}
    for handle, label in zip(handles, labels):
        if not label or label.startswith('_'):
            continue
        if label not in by_label:
            by_label[label] = handle
    return by_label


def _world_legend_items(ax, legend_variant='default'):
    by_label = _legend_handles_by_label(ax)

    if legend_variant == 'compact_rax':
        ordered_entries = [
            ('obstacle', 'Obstacle'),
            ('buffered obstacle', 'Buffered Obstacle'),
            ('obs pts', 'Obstacle Discretization'),
            ('goal', 'Goal'),
            ('start', 'Start'),
            ('vehicle footprint', 'Vehicle Footprint'),
            ('MMR FRS', 'MMR Reach Tube'),
            ('RTD trajectory', 'Safe Trajectory'),
            ('FRS @ k_opt', 'Nominal FRS'),
            ('Nominal FRS', 'Nominal FRS'),
        ]
    else:
        ordered_entries = [
            ('obstacle', 'Obstacle'),
            ('buffered obstacle', 'Buffered Obstacle'),
            ('obs pts', 'Obstacle Discretization'),
            ('goal', 'Goal'),
            ('start', 'Start'),
            ('vehicle footprint', 'Vehicle Footprint'),
            ('robot path', 'Robot Path'),
            ('Immrax collision box', 'True Collision Box'),
            ('MMR FRS', 'MMR FRS'),
            ('RTD trajectory', 'RTD Trajectory'),
            ('FRS @ k_opt', 'Nominal FRS'),
            ('Nominal FRS', 'Nominal FRS'),
        ]

    legend_handles = []
    legend_labels = []
    for old_label, new_label in ordered_entries:
        handle = by_label.get(old_label)
        if handle is None:
            continue
        if old_label == 'goal':
            handle = Line2D(
                [], [], linestyle='None', marker='*',
                color='black', markerfacecolor='black',
                markeredgecolor='black', markersize=11.0,
            )
        elif old_label == 'vehicle footprint':
            facecolor = handle.get_facecolor()
            if np.ndim(facecolor) > 1:
                facecolor = facecolor[0]
            edgecolor = handle.get_edgecolor()
            if np.ndim(edgecolor) > 1:
                edgecolor = edgecolor[0]
            linewidth = handle.get_linewidth()
            if np.ndim(linewidth) > 0:
                linewidth = linewidth[0]
            handle = Line2D(
                [], [], linestyle='None', marker='o',
                markerfacecolor=facecolor, markeredgecolor=edgecolor,
                markeredgewidth=linewidth, markersize=9.5,
                alpha=handle.get_alpha(),
            )
        legend_handles.append(handle)
        legend_labels.append(new_label)
    return legend_handles, legend_labels


def _make_compact_full_gap_figure(
    agent,
    O_upper, O_lower,
    O_buf_upper, O_buf_lower,
    O_pts_upper, O_pts_lower,
    O_FRS, frs, k_opt,
    Z_brk, C_world, cons_poly,
    feasible, frs_version, verify_result=None,
    panel_world_legend='inside',
    hide_result_text=False,
    hide_verify_text=False,
    world_title='World Frame',
    hide_gap_annotation=False,
    overlay_C_world=None,
    overlay_mmr_C_world=None,
    show_goal=False,
    hide_immrax_nominal=False,
    show_start_footprint=False,
    hide_current_footprint=False,
    hide_start_marker=False,
    hide_current_footprint_arrow=False,
    footprint_color_mode='default',
    legend_include_nominal_frs=False,
    legend_include_mmr_frs=False,
    layout='vertical',
    legend_placement='bottom',
):
    show_legend = panel_world_legend != 'none'

    if layout == 'horizontal':
        fig_width = 6.75 if show_legend else 6.45
        fig_height = 3.05 if show_legend else 2.75
        fig, axes = plt.subplots(
            1, 2,
            figsize=(fig_width, fig_height),
            gridspec_kw={'width_ratios': [1.0, 1.15]},
        )
        ax = axes[0]
        ax_world = axes[1]
        k_title_size = 11
        k_label_size = 9
        k_tick_size = 8
        world_font_bump = -16
        world_label_bump = -1
        world_tick_bump = 0
    else:
        fig_height = 5.80 if show_legend else 4.90
        fig, axes = plt.subplots(
            2, 1,
            figsize=(3.45, fig_height),
            gridspec_kw={'height_ratios': [1.0, 1.15]},
        )
        ax = axes[0]
        ax_world = axes[1]
        k_title_size = 12
        k_label_size = 10
        k_tick_size = 8
        world_font_bump = -15
        world_label_bump = 0
        world_tick_bump = -1

    # k-space panel
    ax.set_title('k-space', fontsize=k_title_size)
    ax.set_xlabel('k₂ (speed)', fontsize=k_label_size)
    ax.set_ylabel('k₁ (yaw rate)', fontsize=k_label_size)
    ax.set_aspect('equal')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.45)
    ax.tick_params(
        axis='x', which='both',
        labelsize=k_tick_size, top=False, labeltop=False,
        bottom=True, labelbottom=True,
    )
    ax.tick_params(
        axis='y', which='both',
        labelsize=k_tick_size, right=False, labelright=False,
        left=True, labelleft=True, direction='in',
    )
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    if cons_poly is not None:
        k1g, k2g = np.meshgrid(np.linspace(-1, 1, 60), np.linspace(-1, 1, 60))
        k_flat = np.vstack([k1g.ravel(), k2g.ravel()])
        violated = np.array([
            np.any(eval_constraint_poly(cons_poly, k_flat[:, i]) > 0)
            for i in range(k_flat.shape[1])
        ])
        ax.contourf(
            k2g, k1g, violated.reshape(k2g.shape),
            levels=[0.5, 1.5], colors=[[1.0, 0.5, 0.6]], alpha=0.7,
        )

    if k_opt is not None:
        ax.plot(
            k_opt[1], k_opt[0], 'o',
            color=[0.3, 0.8, 0.5], markersize=8, zorder=5,
        )

    # World panel
    _plot_world(
        ax_world, agent, O_upper, O_lower,
        O_buf_upper, O_buf_lower, O_pts_upper, O_pts_lower,
        Z_brk, C_world, feasible, verify_result,
        world_legend='none',
        minimal_world_text=False,
        hide_result_text=hide_result_text,
        hide_verify_text=hide_verify_text,
        world_title=world_title,
        font_bump=world_font_bump,
        world_label_bump=world_label_bump,
        world_tick_bump=world_tick_bump,
        hide_gap_annotation=hide_gap_annotation,
        overlay_C_world=overlay_C_world,
        overlay_mmr_C_world=overlay_mmr_C_world,
        show_goal=show_goal,
        hide_immrax_nominal=hide_immrax_nominal,
        frs_version=frs_version,
        show_start_footprint=show_start_footprint,
        hide_current_footprint=hide_current_footprint,
        hide_start_marker=hide_start_marker,
        hide_current_footprint_arrow=hide_current_footprint_arrow,
        footprint_color_mode=footprint_color_mode,
        legend_include_nominal_frs=legend_include_nominal_frs,
        legend_include_mmr_frs=legend_include_mmr_frs,
    )
    if layout == 'horizontal':
        ax_world.tick_params(
            axis='x', which='both',
            top=False, labeltop=False,
            bottom=True, labelbottom=True,
        )
        ax_world.tick_params(
            axis='y', which='both',
            right=False, labelright=False,
            left=True, labelleft=True, direction='in',
        )
        ax_world.xaxis.set_ticks_position('bottom')
        ax_world.yaxis.set_ticks_position('left')
        ax_world.yaxis.labelpad = 2.0
    if layout == 'horizontal' and legend_placement == 'right':
        ax_world.yaxis.set_label_coords(-0.115, 0.5)

    legend_handles, legend_labels = _world_legend_items(ax_world, legend_variant='compact_rax')
    if layout == 'horizontal' and legend_placement == 'bottom':
        filtered = [
            (handle, label)
            for handle, label in zip(legend_handles, legend_labels)
            if label != 'Obstacle'
        ]
        legend_handles = [handle for handle, _ in filtered]
        legend_labels = [label for _, label in filtered]
    if show_legend and legend_handles:
        if layout == 'horizontal' and legend_placement == 'right':
            fig.set_size_inches(7.15, 2.75)
            fig.legend(
                legend_handles,
                legend_labels,
                loc='upper left',
                bbox_to_anchor=(0.768, 0.95),
                ncol=1,
                fontsize=7.2,
                frameon=True,
                borderpad=0.28,
                labelspacing=0.24,
                handlelength=1.15,
                handletextpad=0.38,
                columnspacing=0.8,
                markerscale=0.8,
            )
            fig.subplots_adjust(left=0.07, right=0.755, top=0.95, bottom=0.16, wspace=0.24)
        elif layout == 'horizontal' and legend_placement == 'left':
            fig.set_size_inches(7.15, 2.75)
            fig.legend(
                legend_handles,
                legend_labels,
                loc='upper left',
                bbox_to_anchor=(-0.006, 0.95),
                ncol=1,
                fontsize=7.2,
                frameon=True,
                borderpad=0.28,
                labelspacing=0.24,
                handlelength=1.15,
                handletextpad=0.38,
                columnspacing=0.8,
                markerscale=0.8,
            )
            fig.subplots_adjust(left=0.29, right=0.99, top=0.95, bottom=0.16, wspace=0.16)
        else:
            fig.legend(
                legend_handles,
                legend_labels,
                loc='lower center',
                bbox_to_anchor=(0.5, 0.02 if layout == 'vertical' else 0.03),
                ncol=2 if layout == 'vertical' else 5,
                fontsize=7.6 if layout == 'vertical' else 7.55,
                frameon=True,
                borderpad=0.28,
                labelspacing=0.24,
                handlelength=1.2,
                handletextpad=0.4,
                columnspacing=0.8,
                markerscale=0.8,
            )
            if layout == 'horizontal':
                fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.27, wspace=0.00)
            else:
                fig.subplots_adjust(left=0.18, right=0.98, top=0.98, bottom=0.21, hspace=0.42)
    else:
        if layout == 'horizontal':
            fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.12, wspace=0.24)
        else:
            fig.subplots_adjust(left=0.18, right=0.98, top=0.98, bottom=0.11, hspace=0.40)

    return fig


def _plot(agent,
          O_upper, O_lower,
          O_buf_upper, O_buf_lower,
          O_pts_upper, O_pts_lower,
          O_FRS, frs, k_opt,
          T_brk, Z_brk, C_FRS, C_world, cons_poly,
          feasible, frs_version, verify_result=None,
          save_world_fig=None, save_full_fig=None, world_legend='inside',
          panel_world_legend='inside',
          minimal_world_text=False, hide_result_text=False, hide_verify_text=False,
          custom_title=None, world_title='', font_bump=0,
          world_label_bump=0, world_tick_bump=0, hide_v0_in_title=False,
          hide_gap_annotation=False, suptitle_bump=0,
          overlay_C_world=None, overlay_mmr_C_world=None, show_goal=False,
          hide_immrax_nominal=False,
          show_start_footprint=False, hide_current_footprint=False,
          hide_start_marker=False, hide_current_footprint_arrow=False,
          footprint_color_mode='default',
          legend_include_nominal_frs=False, legend_include_mmr_frs=False,
          compact_full_fig=False, compact_full_layout='vertical',
          compact_full_legend_placement='bottom'):

    def fz(size):
        return max(1, size + font_bump)

    status = 'Feasible' if feasible else 'Infeasible'
    if verify_result is not None:
        vsafe = verify_result['safe']
        vstr  = '  Safe' if vsafe else '  Collision'
    else:
        vstr  = ''
    _frs_display = {'standard': 'RTD Standard', 'noerror': 'RTD-RAX'}
    frs_label = _frs_display.get(frs_version, f'RTD {frs_version}')
    suptitle = f'{frs_label}  |  {status}{vstr}'
    if custom_title is not None:
        suptitle = custom_title
    if not hide_v0_in_title:
        suptitle += f'  |  V₀={V_0} m/s'

    if compact_full_fig:
        fig = _make_compact_full_gap_figure(
            agent,
            O_upper, O_lower,
            O_buf_upper, O_buf_lower,
            O_pts_upper, O_pts_lower,
            O_FRS, frs, k_opt,
            Z_brk, C_world, cons_poly,
            feasible, frs_version, verify_result=verify_result,
            panel_world_legend=panel_world_legend,
            hide_result_text=hide_result_text,
            hide_verify_text=hide_verify_text,
            world_title=world_title or 'World Frame',
            hide_gap_annotation=hide_gap_annotation,
            overlay_C_world=overlay_C_world,
            overlay_mmr_C_world=overlay_mmr_C_world,
            show_goal=show_goal,
            hide_immrax_nominal=hide_immrax_nominal,
            show_start_footprint=show_start_footprint,
            hide_current_footprint=hide_current_footprint,
            hide_start_marker=hide_start_marker,
            hide_current_footprint_arrow=hide_current_footprint_arrow,
            footprint_color_mode=footprint_color_mode,
            legend_include_nominal_frs=legend_include_nominal_frs,
            legend_include_mmr_frs=legend_include_mmr_frs,
            layout=compact_full_layout,
            legend_placement=compact_full_legend_placement,
        )
    else:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(suptitle, fontsize=fz(27 + suptitle_bump),
                     color='green' if feasible else 'red',
                     y=0.98 if panel_world_legend != 'outside_top' else 1.0)

        # ---- Panel 1: k-space -----------------------------------------------
        ax = axes[0]
        ax.set_title('k-space', fontsize=fz(26))
        ax.set_xlabel('k₂ (speed)', fontsize=fz(25))
        ax.set_ylabel('k₁ (yaw rate)', fontsize=fz(25))
        ax.set_aspect('equal')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True)
        ax.tick_params(axis='both', labelsize=fz(24))

        if cons_poly is not None:
            k1g, k2g = np.meshgrid(np.linspace(-1, 1, 60), np.linspace(-1, 1, 60))
            k_flat   = np.vstack([k1g.ravel(), k2g.ravel()])
            violated = np.array([
                np.any(eval_constraint_poly(cons_poly, k_flat[:, i]) > 0)
                for i in range(k_flat.shape[1])
            ])
            ax.contourf(k2g, k1g, violated.reshape(k2g.shape),
                        levels=[0.5, 1.5], colors=[[1.0, 0.5, 0.6]], alpha=0.7)

        if k_opt is not None:
            ax.plot(k_opt[1], k_opt[0], 'o', color=[0.3, 0.8, 0.5],
                    markersize=13, zorder=5, label='k_opt')
            if panel_world_legend not in ('none', 'outside_top'):
                ax.legend(fontsize=fz(15))

        # ---- Panel 2: FRS frame ----------------------------------------------
        ax = axes[1]
        ax.set_title('FRS Frame', fontsize=fz(26))
        ax.set_xlabel('z₁ (scaled x)', fontsize=fz(25))
        ax.set_ylabel('z₂ (scaled y)', fontsize=fz(25))
        ax.set_aspect('equal')
        ax.grid(True)
        ax.tick_params(axis='both', labelsize=fz(24))

        if frs.get('pows_hZ0') is not None:
            z1g, z2g = np.meshgrid(np.linspace(-1.1, 1.1, 50),
                                    np.linspace(-1.1, 1.1, 50))
            z_grid   = np.vstack([z1g.ravel(), z2g.ravel()])
            zp_h     = frs['pows_hZ0'][:, frs['z_cols_hZ0']]
            zv_h     = np.prod(z_grid[np.newaxis, :, :] ** zp_h[:, :, np.newaxis], axis=1)
            h_z0     = frs['coef_hZ0'] @ zv_h
            ax.contour(z1g, z2g, h_z0.reshape(z1g.shape), levels=[0.0],
                       colors='blue', linewidths=1.8)
            ax.plot([], [], color='blue', linewidth=1.8, label='FRS boundary')

        if O_FRS.shape[1] > 0:
            ax.plot(O_FRS[0], O_FRS[1], '.', color=[0.5, 0.1, 0.1],
                    markersize=7, label='obs pts')

        if C_FRS is not None:
            ax.plot(C_FRS[0], C_FRS[1], color=[0.3, 0.8, 0.5],
                    linewidth=1.8, label='FRS @ k_opt')

        if panel_world_legend not in ('none', 'outside_top'):
            ax.legend(fontsize=fz(15))

        # ---- Panel 3: world frame --------------------------------------------
        _plot_world(axes[2], agent, O_upper, O_lower,
                    O_buf_upper, O_buf_lower, O_pts_upper, O_pts_lower,
                    Z_brk, C_world, feasible, verify_result,
                    world_legend=panel_world_legend,
                    minimal_world_text=minimal_world_text,
                    hide_result_text=hide_result_text,
                    hide_verify_text=hide_verify_text,
                    world_title=world_title,
                    font_bump=font_bump,
                    world_label_bump=world_label_bump,
                    world_tick_bump=world_tick_bump,
                    hide_gap_annotation=hide_gap_annotation,
                    overlay_C_world=overlay_C_world,
                    overlay_mmr_C_world=overlay_mmr_C_world,
                    show_goal=show_goal,
                    hide_immrax_nominal=hide_immrax_nominal,
                    frs_version=frs_version,
                    show_start_footprint=show_start_footprint,
                    hide_current_footprint=hide_current_footprint,
                    hide_start_marker=hide_start_marker,
                    hide_current_footprint_arrow=hide_current_footprint_arrow,
                    footprint_color_mode=footprint_color_mode,
                    legend_include_nominal_frs=legend_include_nominal_frs,
                    legend_include_mmr_frs=legend_include_mmr_frs)
        fig.tight_layout()
    if save_full_fig:
        fig.savefig(save_full_fig, bbox_inches='tight', pad_inches=0.02)
        print(f'Saved full figure to: {save_full_fig}')

    # ---- Separate world-frame figure -------------------------------------
    if world_legend == 'outside':
        world_figsize = (11.0, 6.0) if verify_result is not None else (8.8, 6.0)
    else:
        world_figsize = (6, 6)
    fig2, ax2 = plt.subplots(figsize=world_figsize)
    _plot_world(ax2, agent, O_upper, O_lower,
                O_buf_upper, O_buf_lower, O_pts_upper, O_pts_lower,
                Z_brk, C_world, feasible, verify_result,
                world_legend=world_legend,
                minimal_world_text=minimal_world_text,
                hide_result_text=hide_result_text,
                hide_verify_text=hide_verify_text,
                world_title=world_title,
                font_bump=font_bump,
                world_label_bump=world_label_bump,
                world_tick_bump=world_tick_bump,
                hide_gap_annotation=hide_gap_annotation,
                overlay_C_world=overlay_C_world,
                overlay_mmr_C_world=overlay_mmr_C_world,
                show_goal=show_goal,
                hide_immrax_nominal=hide_immrax_nominal,
                frs_version=frs_version,
                show_start_footprint=show_start_footprint,
                hide_current_footprint=hide_current_footprint,
                hide_start_marker=hide_start_marker,
                hide_current_footprint_arrow=hide_current_footprint_arrow,
                footprint_color_mode=footprint_color_mode,
                legend_include_nominal_frs=legend_include_nominal_frs,
                legend_include_mmr_frs=legend_include_mmr_frs)
    fig2.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    # Center suptitle over axes (not figure) so bbox_inches='tight' doesn't skew it
    ax_pos = ax2.get_position()
    fig2.suptitle(suptitle, fontsize=fz(24 + suptitle_bump),
                  color='green' if feasible else 'red',
                  x=(ax_pos.x0 + ax_pos.x1) / 2.0)
    if save_world_fig:
        fig2.savefig(save_world_fig, bbox_inches='tight', pad_inches=0.02)
        print(f'Saved world-frame figure to: {save_world_fig}')


def _plot_world(ax, agent,
                O_upper, O_lower,
                O_buf_upper, O_buf_lower,
                O_pts_upper, O_pts_lower,
                Z_brk, C_world, feasible,
                verify_result=None,
                world_legend='inside',
                minimal_world_text=False,
                hide_result_text=False,
                hide_verify_text=False,
                world_title='',
                font_bump=0,
                world_label_bump=0,
                world_tick_bump=0,
                hide_gap_annotation=False,
                overlay_C_world=None,
                overlay_mmr_C_world=None,
                show_goal=False,
                hide_immrax_nominal=False,
                frs_version='standard',
                show_start_footprint=False,
                hide_current_footprint=False,
                hide_start_marker=False,
                hide_current_footprint_arrow=False,
                footprint_color_mode='default',
                legend_include_nominal_frs=False,
                legend_include_mmr_frs=False):

    def fz(size):
        return max(1, size + font_bump)

    ax.set_title(world_title, fontsize=fz(27))
    ax.set_xlabel('x [m]', fontsize=fz(25 + world_label_bump))
    ax.set_ylabel('y [m]', fontsize=fz(25 + world_label_bump))
    ax.set_aspect('equal')
    ax.tick_params(axis='both', labelsize=fz(24 + world_tick_bump))

    half_gap  = GAP_WIDTH / 2.0
    y_extent  = half_gap + OBS_HEIGHT + 0.15
    x_lo = -0.3
    x_hi = OBS_X + OBS_HALF_WIDTH + 0.3
    if show_goal:
        x_hi = max(x_hi, X_DES + 0.2)
    x_range = x_hi - x_lo
    y_range = 2.0 * y_extent
    # Widen x to match y range so aspect='equal' gives same height as square panels
    if x_range < y_range:
        x_mid = (x_lo + x_hi) / 2.0
        x_lo = x_mid - y_range / 2.0
        x_hi = x_mid + y_range / 2.0
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-y_extent, y_extent)

    obs_color     = [0.75, 0.25, 0.25]
    buf_color     = [1.0,  0.55, 0.55]
    pts_color     = [0.4,  0.05, 0.05]

    for O, O_buf, O_pts in [(O_upper, O_buf_upper, O_pts_upper),
                             (O_lower, O_buf_lower, O_pts_lower)]:
        ax.fill(O[0], O[1], color=obs_color, alpha=0.9, label='_nolegend_')
        if O_buf is not None and O_buf.shape[1] > 0:
            ax.fill(O_buf[0], O_buf[1], color=buf_color, alpha=0.5,
                    label='_nolegend_')
        if O_pts is not None and O_pts.shape[1] > 0:
            ax.plot(O_pts[0], O_pts[1], '.', color=pts_color,
                    markersize=5, label='_nolegend_')

    # Legend proxies for the two obstacle layers
    ax.fill([], [], color=obs_color, alpha=0.9, label='obstacle')
    ax.fill([], [], color=buf_color, alpha=0.5, label='buffered obstacle')
    ax.plot([], [], '.', color=pts_color, markersize=5, label='obs pts')

    # Gap annotation
    if not minimal_world_text and not hide_gap_annotation:
        ax.axhline( half_gap, color='gray', linestyle=':', linewidth=0.8)
        ax.axhline(-half_gap, color='gray', linestyle=':', linewidth=0.8)
        ax.annotate('', xy=(OBS_X + OBS_HALF_WIDTH + 0.15,  half_gap),
                    xytext=(OBS_X + OBS_HALF_WIDTH + 0.15, -half_gap),
                    arrowprops=dict(arrowstyle='<->', color='gray'))
        ax.text(OBS_X + OBS_HALF_WIDTH + 0.18, 0,
                f'gap\n{GAP_WIDTH:.2f} m', fontsize=fz(22), va='center', color='gray')

    # Goal
    ax.plot(X_DES, Y_DES, 'k*', markersize=17, label='goal')

    # Trajectory
    if Z_brk is not None:
        ax.plot(Z_brk[0], Z_brk[1], color='purple', linestyle='--',
                linewidth=2.5, alpha=0.95, zorder=7, label='RTD trajectory')

    # FRS contour
    if C_world is not None:
        ax.plot(C_world[0], C_world[1], color=[0.3, 0.8, 0.5],
                linewidth=1.8, label='FRS @ k_opt')

    # Overlay noerror FRS contour
    if overlay_C_world is not None:
        ax.plot(overlay_C_world[0], overlay_C_world[1], color=[0.3, 0.8, 0.5],
                linewidth=1.8, label='Nominal FRS')

    # Overlay MMR (standard/tracking-error) FRS contour at same k
    if overlay_mmr_C_world is not None:
        ax.plot(overlay_mmr_C_world[0], overlay_mmr_C_world[1], color=[0.2, 0.4, 0.85],
                linewidth=1.8, linestyle='--', label='MMR FRS')

    # Legend proxy entries for items not plotted on this panel
    if legend_include_nominal_frs:
        ax.plot([], [], color=[0.3, 0.8, 0.5], linewidth=1.8, label='Nominal FRS')
    if legend_include_mmr_frs:
        ax.plot([], [], color=[0.2, 0.4, 0.85], linewidth=1.8, linestyle='--', label='MMR FRS')

    # Robot
    compare_footprint_colors = {
        'standard': '#8e1f1f',
        'noerror': '#1f6f3d',
    }
    footprint_color = 'steelblue'
    footprint_alpha = 0.5
    footprint_edgecolor = None
    if footprint_color_mode == 'compare':
        footprint_color = compare_footprint_colors.get(frs_version, footprint_color)
        footprint_alpha = 0.32
        footprint_edgecolor = 'black'

    st = np.asarray(agent.state)
    if not hide_start_marker:
        ax.plot(st[0, 0], st[1, 0], marker='o', markerfacecolor='none',
                markeredgecolor='black', markersize=9, linestyle='None',
                label='start', zorder=7)
    if st.shape[1] > 1:
        ax.plot(st[0, :], st[1, :],
                color='steelblue', linewidth=1.5, zorder=1, label='robot path')
    ax.fill([], [], color=footprint_color, alpha=footprint_alpha, label='vehicle footprint')

    if not hide_result_text:
        result_str = 'PATH FOUND' if feasible else 'NO PATH FOUND'
        result_color = 'green' if feasible else 'red'
        ax.text(0.02, 0.97, result_str, transform=ax.transAxes,
            fontsize=fz(26), fontweight='bold', color=result_color,
            va='top')

    # --- immrax reachable tube and verdict ---
    if verify_result is not None:
        xy_tube = verify_result['xy_tube']   # (N, 4) [x_lo, x_hi, y_lo, y_hi]
        ts_tube = np.asarray(verify_result.get('ts_tube', np.arange(len(xy_tube))), dtype=float)
        xy_swept = verify_result.get('xy_swept', np.zeros((0, 4)))
        nom_xy  = verify_result['nom_xy']    # (M, 2) [x, y]
        vsafe   = verify_result['safe']
        expanded_obs = verify_result.get('expanded_obs', [])
        tube_cmap_name = 'Blues' if vsafe else 'Reds'
        tube_cmap = plt.get_cmap(tube_cmap_name)
        nom_col  = [0.98, 0.65, 0.10] if vsafe else [0.20, 0.90, 0.95]

        # Draw reachable tube rectangles with time-gradient coloring.
        import matplotlib.patches as _mp
        n_tube = max(len(xy_tube), 1)
        for i, row in enumerate(xy_tube):
            x_lo, x_hi, y_lo, y_hi = row
            frac = i / max(n_tube - 1, 1)
            tube_col = tube_cmap(0.30 + 0.60 * frac)
            rect = _mp.FancyBboxPatch(
                (x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                boxstyle='square,pad=0',
                linewidth=0,
                facecolor=tube_col, alpha=0.12,
                zorder=3,
            )
            ax.add_patch(rect)

        # Draw swept hull rectangles (between samples) to match collision math.
        n_swept = max(len(xy_swept), 1)
        for i, row in enumerate(xy_swept):
            x_lo, x_hi, y_lo, y_hi = row
            frac = i / max(n_swept - 1, 1)
            swept_col = tube_cmap(0.25 + 0.65 * frac)
            rect = _mp.FancyBboxPatch(
                (x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                boxstyle='square,pad=0',
                linewidth=0,
                facecolor=swept_col, alpha=0.06,
                zorder=2,
            )
            ax.add_patch(rect)

        # Plot per-axis tube bounds as thin lines to show interval envelope.
        if len(xy_tube) > 1:
            line_col = tube_cmap(0.92)
            x_mid = 0.5 * (xy_tube[:, 0] + xy_tube[:, 1])
            y_mid = 0.5 * (xy_tube[:, 2] + xy_tube[:, 3])
            ax.plot(x_mid, xy_tube[:, 2], color=line_col, linewidth=0.8, alpha=0.8, zorder=4)
            ax.plot(x_mid, xy_tube[:, 3], color=line_col, linewidth=0.8, alpha=0.8, zorder=4)
            ax.plot(xy_tube[:, 0], y_mid, color=line_col, linewidth=0.8, alpha=0.8, zorder=4)
            ax.plot(xy_tube[:, 1], y_mid, color=line_col, linewidth=0.8, alpha=0.8, zorder=4)

        # Draw robot-inflated obstacle boxes used by immrax collision checks.
        for i, (ox_lo, ox_hi, oy_lo, oy_hi) in enumerate(expanded_obs):
            rect = _mp.Rectangle(
                (ox_lo, oy_lo), ox_hi - ox_lo, oy_hi - oy_lo,
                fill=False, linewidth=1.2, linestyle='--',
                edgecolor='black', alpha=0.7, zorder=5,
                label='Immrax collision box' if i == 0 else '_nolegend_',
            )
            ax.add_patch(rect)

        # Legend proxy + nominal trajectory in contrasting color.
        if len(xy_tube) > 0:
            ax.plot([], [], color=tube_cmap(0.92), alpha=0.9, linewidth=2.0,
                    label='MMR FRS')
            if not hide_immrax_nominal:
                ax.plot(nom_xy[:, 0], nom_xy[:, 1],
                        color=nom_col, linewidth=2.5, linestyle='--',
                        alpha=1.0, label='Immrax nominal', zorder=4)

            # Arrow to show time direction along tube sampling.
            i0 = 0
            i1 = min(max(len(xy_tube) - 1, 0), 4) if len(xy_tube) > 4 else len(xy_tube) - 1
            if i1 > i0:
                p0 = np.array([0.5 * (xy_tube[i0, 0] + xy_tube[i0, 1]),
                               0.5 * (xy_tube[i0, 2] + xy_tube[i0, 3])])
                p1 = np.array([0.5 * (xy_tube[i1, 0] + xy_tube[i1, 1]),
                               0.5 * (xy_tube[i1, 2] + xy_tube[i1, 3])])
                ax.annotate('', xy=p1, xytext=p0,
                            arrowprops=dict(arrowstyle='->', color=tube_cmap(0.95), lw=1.2))

        if not minimal_world_text and not hide_verify_text:
            vtxt   = 'immrax: SAFE' if vsafe else 'immrax: COLLISION'
            vcolor = 'steelblue'     if vsafe else 'orangered'
            ax.text(0.02, 0.88, vtxt, transform=ax.transAxes,
                    fontsize=fz(24), fontweight='bold', color=vcolor, va='top')

            if not vsafe and verify_result['collision_time'] is not None:
                tc = verify_result['collision_time']
                ax.text(0.02, 0.82, f't = {tc:.2f} s', transform=ax.transAxes,
                        fontsize=fz(23), color=vcolor, va='top')

    start_x = float(st[0, 0])
    start_y = float(st[1, 0])
    start_h = float(st[2, 0])
    current_x = float(st[0, -1])
    current_y = float(st[1, -1])
    current_h = float(st[2, -1])
    moved_from_start = np.hypot(current_x - start_x, current_y - start_y) > 1e-9

    if show_start_footprint:
        _draw_vehicle_icon(
            ax,
            start_x,
            start_y,
            start_h,
            agent.footprint,
            color=footprint_color,
            alpha=footprint_alpha,
            show_center_ring=False,
            show_arrow=True,
            edgecolor=footprint_edgecolor,
            linewidth=1.0,
        )

    if not hide_current_footprint:
        _draw_vehicle_icon(
            ax,
            current_x,
            current_y,
            current_h,
            agent.footprint,
            color=footprint_color,
            alpha=footprint_alpha,
            show_center_ring=moved_from_start,
            show_arrow=not hide_current_footprint_arrow,
            edgecolor=footprint_edgecolor,
            linewidth=1.0,
        )

    legend_handles, legend_labels = _world_legend_items(ax)
    outside_world_legend = (world_legend == 'outside')
    outside_top_legend   = (world_legend == 'outside_top')
    if outside_top_legend:
        legend_fontsize = max(8, min(19, fz(16)))
    elif outside_world_legend:
        legend_fontsize = max(8, min(19, fz(15)))
    else:
        legend_fontsize = max(8, min(19, fz(11)))
    legend_kwargs = dict(
        fontsize=legend_fontsize,
        frameon=True,
        borderaxespad=0.0,
        borderpad=0.28 if (outside_world_legend or outside_top_legend) else 0.35,
        labelspacing=0.22 if outside_top_legend else (0.28 if outside_world_legend else 0.35),
        handlelength=1.2 if outside_top_legend else (1.35 if outside_world_legend else 1.5),
        handletextpad=0.35 if outside_top_legend else 0.45,
        markerscale=0.7 if outside_top_legend else (0.8 if outside_world_legend else 0.9),
    )

    outside_anchor_x = 1.16 if (world_legend == 'outside' and verify_result is not None) else 1.04
    outside_anchor_y = 1.08

    if world_legend == 'outside':
        if legend_handles:
            ax.legend(legend_handles, legend_labels,
                      loc='upper left', bbox_to_anchor=(outside_anchor_x, outside_anchor_y),
                      **legend_kwargs)
    elif world_legend == 'outside_top':
        if legend_handles:
            ax.legend(legend_handles, legend_labels,
                      loc='upper left', bbox_to_anchor=(1.04, 1.0),
                      **legend_kwargs)
    elif world_legend == 'inside':
        if legend_handles:
            ax.legend(legend_handles, legend_labels,
                      loc='lower right', **legend_kwargs)


# ===========================================================================

if __name__ == '__main__':
    main()
