"""
one_shot_rtd.py  –  NumPy / SciPy version
============================================
One-shot RTD trajectory optimisation for the Turtlebot.

Python translation of:
  RTD_tutorial/step_4_online_planning/scripts/step_4_one_shot_turtlebot_RTD.m

Flow:
  1. Set initial speed and desired goal position
  2. Load FRS (requires *_preproc.mat from preprocess_frs.m)
  3. Create agent, place random obstacle
  4. Build cost + constraint functions
  5. Run scipy.optimize.minimize (SLSQP)
  6. Move agent along optimal braking trajectory
  7. Plot: world frame  +  FRS frame  +  k-space
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')   # change to 'Qt5Agg' or 'Agg' if TkAgg unavailable
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize, Bounds

sys.path.insert(0, os.path.dirname(__file__))

from frs_loader        import load_frs, k_to_wv
from geometry_utils    import (world_to_local, FRS_to_world,
                                compute_turtlebot_point_spacing,
                                compute_turtlebot_discretized_obs,
                                make_random_polygon)
from trajectory        import make_turtlebot_braking_trajectory
from turtlebot_agent   import TurtlebotAgent
from polynomial_utils  import (get_frs_polynomial_structure,
                                evaluate_frs_polynomial_on_obstacle_points,
                                get_constraint_polynomial_gradient,
                                eval_constraint_poly)
from cost              import turtlebot_cost_and_grad
from constraints       import build_constraint


# ===========================================================================
# User parameters
# ===========================================================================

V_0             = 0.5    # initial speed (m/s)  in [0.0, 1.5]
X_DES           = 0.75   # desired x  (world frame, m)
Y_DES           = 0.5    # desired y  (world frame, m)

N_VERTICES      = 5
OBSTACLE_SCALE  = 0.3
OBSTACLE_BUFFER = 0.05   # m

RANDOM_SEED     = None   # set an int for reproducibility


# ===========================================================================
# Main
# ===========================================================================

def main():
    rng = np.random.default_rng(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # --- Load FRS ---
    print('Loading FRS...')
    frs = load_frs(v_0=V_0)

    # --- Create agent ---
    agent = TurtlebotAgent()
    agent.reset([0.0, 0.0, 0.0, V_0])

    # --- Random obstacle ---
    obs_x = 0.5 + rng.random() * 0.75
    obs_y = (rng.random() - 0.5) * 0.8
    O = make_random_polygon(N_VERTICES, [obs_x, obs_y], OBSTACLE_SCALE)
    print(f'Obstacle at ({obs_x:.3f}, {obs_y:.3f})')

    # --- Goal in robot-local frame ---
    z_goal       = np.array([[X_DES], [Y_DES]])
    z_goal_local = np.asarray(world_to_local(agent.state[:, -1], z_goal)).reshape(-1)
    x_des_loc    = float(z_goal_local[0])
    y_des_loc    = float(z_goal_local[1])

    # --- Discretise obstacle → FRS frame ---
    r = compute_turtlebot_point_spacing(agent.footprint, OBSTACLE_BUFFER)
    O_FRS, O_buf, O_pts = compute_turtlebot_discretized_obs(
        O, agent.state[:, -1], OBSTACLE_BUFFER, r, frs
    )

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
        T_brk, U_brk, Z_brk = make_turtlebot_braking_trajectory(t_plan, t_stop, w_des, v_des)
        agent.move(T_brk[-1], T_brk, U_brk, Z_brk)
        print(f'Agent moved.  Final pose: {agent.pose}')

    # --- FRS contour in z-space (evaluate FRS poly on a grid, sub k=k_opt) ---
    C_FRS = C_world = None
    if k_opt is not None:
        C_FRS, C_world = _compute_frs_contour(frs, k_opt, agent.state[:, 0])

    # --- Plot ---
    _plot(agent, O, O_buf, O_pts, O_FRS, frs, k_opt,
          T_brk, Z_brk, C_FRS, C_world, cons_poly)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_frs_contour(frs, k_opt, initial_pose, grid_res=100):
    """Evaluate the FRS polynomial at k_opt on a z-grid; extract the contour."""
    z1g, z2g = np.meshgrid(np.linspace(-1, 1, grid_res),
                            np.linspace(-1, 1, grid_res))
    z_grid  = np.vstack([z1g.ravel(), z2g.ravel()])   # (2, Ngrid)

    pows   = frs['pows']                               # (Nterms, Nvars)
    coef   = frs['coef']                               # (Nterms,)
    z_cols = frs['z_cols']
    k_cols = frs['k_cols']

    k_pows  = pows[:, k_cols]                          # (Nterms, Nk)
    z_pows  = pows[:, z_cols]                          # (Nterms, Nz)
    k_mono  = np.prod(k_opt[np.newaxis, :] ** k_pows, axis=1)   # (Nterms,)
    z_vals  = np.prod(z_grid[np.newaxis, :, :] ** z_pows[:, :, np.newaxis], axis=1)  # (Nterms, Ngrid)
    frs_grid = (coef * k_mono) @ z_vals                # (Ngrid,)  [= FRS_poly(z,k) - 1]

    # Extract contour at level 0  (FRS_poly = 1  ↔  FRS boundary)
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(z1g, z2g, frs_grid.reshape(z1g.shape), levels=[0.0])
    plt.close(fig_tmp)

    C_FRS = C_world = None
    if cs.allsegs and cs.allsegs[0]:
        # Keep only closed contour loops (the FRS boundary); this removes
        # open/spurious segments that can appear at the grid boundary.
        segs_all = cs.allsegs[0]
        close_tol = 5.0 / float(grid_res)
        segs = [
            seg for seg in segs_all
            if seg.size > 0 and np.linalg.norm(seg[0] - seg[-1]) <= close_tol
        ]
        if not segs:
            segs = segs_all

        # Keep all remaining disconnected contours with NaN separators.
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


def _plot(agent, O, O_buf, O_pts, O_FRS, frs,
          k_opt, T_brk, Z_brk, C_FRS, C_world, cons_poly):

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('One-shot Turtlebot RTD  –  NumPy/SciPy', fontsize=14)

    # ---- Panel 1: k-space -----------------------------------------------
    ax = axes[0]
    ax.set_title('Traj Params (k-space)')
    ax.set_xlabel('k₂ (speed param)')
    ax.set_ylabel('k₁ (yaw-rate param)')
    ax.set_aspect('equal')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True)

    if cons_poly is not None:
        # Brute-force grid: identify violated region
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
                markersize=12, zorder=5, label='k_opt')
        ax.legend()

    # ---- Panel 2: FRS frame ----------------------------------------------
    ax = axes[1]
    ax.set_title('FRS Frame')
    ax.set_xlabel('z₁ (scaled x)')
    ax.set_ylabel('z₂ (scaled y)')
    ax.set_aspect('equal')
    ax.grid(True)

    if frs.get('pows_hZ0') is not None:
        z1g, z2g = np.meshgrid(np.linspace(-1.1, 1.1, 50),
                                np.linspace(-1.1, 1.1, 50))
        z_grid   = np.vstack([z1g.ravel(), z2g.ravel()])
        zp_h     = frs['pows_hZ0'][:, frs['z_cols_hZ0']]  # (Nt, Nz)
        zv_h     = np.prod(z_grid[np.newaxis, :, :] ** zp_h[:, :, np.newaxis], axis=1)
        h_z0     = frs['coef_hZ0'] @ zv_h
        ax.contour(z1g, z2g, h_z0.reshape(z1g.shape), levels=[0.0],
                   colors='blue', linewidths=1.5)
        ax.plot([], [], color='blue', linewidth=1.5, label='FRS boundary')

    if O_FRS is not None and O_FRS.shape[1] > 0:
        ax.plot(O_FRS[0], O_FRS[1], '.', color=[0.5, 0.1, 0.1], markersize=8,
                label='obs pts')

    if C_FRS is not None:
        ax.plot(C_FRS[0], C_FRS[1], color=[0.3, 0.8, 0.5], linewidth=1.5,
                label='FRS @ k_opt')

    ax.legend(fontsize=8)

    # ---- Panel 3: world frame --------------------------------------------
    _plot_world_frame(axes[2], agent, O, O_buf, O_pts, Z_brk, C_world)

    # ---- Separate world-frame figure -------------------------------------
    fig_world, ax_world = plt.subplots(1, 1, figsize=(6, 5))
    fig_world.suptitle('World Frame Only  –  NumPy/SciPy', fontsize=12)
    _plot_world_frame(ax_world, agent, O, O_buf, O_pts, Z_brk, C_world)
    fig_world.tight_layout()


def _plot_world_frame(ax, agent, O, O_buf, O_pts, Z_brk, C_world):
    ax.set_title('World Frame')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-1.0, 1.0)

    if O_buf is not None and O_buf.shape[1] > 0:
        ax.fill(O_buf[0], O_buf[1], color=[1.0, 0.5, 0.6], alpha=0.7,
                label='buffered obs')
    if O.shape[1] > 0:
        ax.fill(O[0], O[1], color=[1.0, 0.7, 0.8], label='obstacle')
    if O_pts is not None and O_pts.shape[1] > 0:
        ax.plot(O_pts[0], O_pts[1], '.', color=[0.5, 0.1, 0.1], markersize=6,
                label='obs pts')

    ax.plot(X_DES, Y_DES, 'k*', markersize=15, linewidth=2, label='goal')

    if Z_brk is not None:
        ax.plot(Z_brk[0], Z_brk[1], 'b--', linewidth=1.5, label='trajectory')

    if C_world is not None:
        ax.plot(C_world[0], C_world[1], color=[0.3, 0.8, 0.5],
                linewidth=1.5, label='FRS @ k_opt')

    st = np.asarray(agent.state)
    ax.plot(st[0, 0], st[1, 0], marker='o', markerfacecolor='none',
            markeredgecolor='black', markersize=7, linestyle='None',
            label='start pose')
    ax.plot([], [], color='steelblue', linewidth=1.5, label='robot path')
    ax.plot([], [], color='steelblue', linewidth=8, alpha=0.4,
            label='robot footprint (current)')
    agent.plot(ax=ax, color='steelblue', alpha=0.4)
    ax.legend(fontsize=8, loc='upper left')


# ===========================================================================

if __name__ == '__main__':
    main()
