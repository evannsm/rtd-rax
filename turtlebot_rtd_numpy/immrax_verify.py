"""
immrax_verify.py  –  immrax reachability verification for Turtlebot RTD
========================================================================
Given RTD-planned braking trajectory parameters, compute the reachable
tube of the closed-loop turtlebot and check for obstacle intersection.

The braking control law is computed analytically from time t, so no
table-interpolation primitive needs to be registered with immrax's NIF.
All operations used (arithmetic, sin/cos, min/max, clip) are already in
immrax's inclusion_registry as of v0.3.x.

Usage
-----
    from immrax_verify import verify

    result = verify(
        w_des, v_des, t_plan, t_stop,
        z0        = [0., 0., 0., 0.75],
        obs_rects = [(x_lo, x_hi, y_lo, y_hi), ...],
        robot_radius    = 0.175,   # m
        init_uncertainty = 0.01,   # ±epsilon ball on initial state
    )
    print(result['safe'])          # bool
    print(result['xy_tube'])       # (N, 4) array [x_lo, x_hi, y_lo, y_hi]

References
----------
  github.com/gtfactslab/immrax
  RTD_combo demo: edgar_DEMOVersionwithNewImmrax.ipynb
"""

import numpy as np
import jax.numpy as jnp
import immrax as irx

try:
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.geometry import box as shapely_box
    _SHAPELY = True
except ImportError:
    _SHAPELY = False


def _normalize_disturbance_bound(disturbance_bound):
    """Return a nonnegative length-4 disturbance half-width vector."""
    if np.isscalar(disturbance_bound):
        d = max(float(disturbance_bound), 0.0) * np.ones(4, dtype=float)
    else:
        d = np.asarray(disturbance_bound, dtype=float).ravel()
        if d.shape != (4,):
            raise ValueError('disturbance_bound must be a scalar or length-4 vector')
        d = np.maximum(d, 0.0)
    return d


def _normalize_disturbance_interval(disturbance_interval=None, disturbance_bound: float = 0.0):
    """Return signed disturbance bounds as (lower, upper) length-4 vectors."""
    if disturbance_interval is None:
        d = _normalize_disturbance_bound(disturbance_bound)
        return -d, d

    if isinstance(disturbance_interval, (tuple, list)) and len(disturbance_interval) == 2:
        d_lo = np.asarray(disturbance_interval[0], dtype=float).ravel()
        d_hi = np.asarray(disturbance_interval[1], dtype=float).ravel()
    else:
        arr = np.asarray(disturbance_interval, dtype=float)
        if arr.shape == (2, 4):
            d_lo = arr[0, :]
            d_hi = arr[1, :]
        elif arr.shape == (4, 2):
            d_lo = arr[:, 0]
            d_hi = arr[:, 1]
        else:
            raise ValueError('disturbance_interval must have shape (2, 4), (4, 2), or be a pair of length-4 vectors')

    if d_lo.shape != (4,) or d_hi.shape != (4,):
        raise ValueError('disturbance_interval must provide lower/upper length-4 vectors')
    return np.minimum(d_lo, d_hi), np.maximum(d_lo, d_hi)


# ---------------------------------------------------------------------------
# Turtlebot system with analytically-computed braking control
# ---------------------------------------------------------------------------

class TurtleBotBraking(irx.System):
    """Unicycle turtlebot tracking a braking trajectory (open-loop control).

    The planned braking trajectory has:
      - cruise phase   t ∈ [0, t_plan]:             w = w_des,  a = 0
      - braking phase  t ∈ [t_plan, t_plan+t_stop]: w and v scale linearly to 0

    State  x = [px, py, heading, v, w_des, v_des, t_plan, t_stop, dpx, dpy, dh, dv]
    Input  w = [dpx, dpy, dh, dv]  (external disturbance, zero for verification)

    The final four state entries encode the disturbance-interval center. Their
    interval radii inside the initial set encode the disturbance half-widths.

    The control is baked in analytically so that immrax's NIF (natif) never
    encounters a non-registered primitive such as jnp.interp.
    """

    def __init__(self):
        super().__init__('continuous', 12)

    # -- helpers (only use registered ops) -----------------------------------

    def _braking_scale(self, t, t_plan, t_stop):
        """Braking scale ∈ [0, 1]: 1 during cruise, linear decay after t_plan."""
        # Registered ops: sub, div, min_p, max_p (passthrough in immrax nif.py)
        safe_t_stop = jnp.maximum(t_stop, jnp.float32(1e-3))
        scale_raw = (t_plan + safe_t_stop - t) / safe_t_stop
        return jnp.minimum(jnp.maximum(scale_raw, jnp.float32(0.0)),
                           jnp.float32(1.0))

    def _accel_gate(self, t, t_plan, t_stop):
        """Smooth 0/1 gate for t in [t_plan, t_plan + t_stop] using clip only."""
        eps = jnp.float32(1e-4)
        safe_t_stop = jnp.maximum(t_stop, jnp.float32(1e-3))
        rise = jnp.clip((t - t_plan) / eps, jnp.float32(0.0), jnp.float32(1.0))
        fall = jnp.clip((t_plan + safe_t_stop - t) / eps,
                        jnp.float32(0.0), jnp.float32(1.0))
        return rise * fall

    def _accel_cmd(self, t, v_des, t_plan, t_stop):
        """Commanded acceleration: ~0 during cruise, ~-v_des/t_stop during braking."""
        safe_t_stop = jnp.maximum(t_stop, jnp.float32(1e-3))
        return (-v_des / safe_t_stop) * self._accel_gate(t, t_plan, safe_t_stop)

    # -- dynamics ------------------------------------------------------------

    def f(self, t, x, w):
        """ẋ = f(t, x, w) — unicycle with time-varying braking control."""
        h = x[2]
        v = x[3]
        w_des = x[4]
        v_des = x[5]
        t_plan = x[6]
        t_stop = x[7]

        scale = self._braking_scale(t, t_plan, t_stop)
        w_cmd = w_des * scale
        a_cmd = self._accel_cmd(t, v_des, t_plan, t_stop)

        # Registered: sin_p, cos_p, mul_p, add_p
        xd = v * jnp.cos(h) + w[0]
        yd = v * jnp.sin(h) + w[1]
        hd = w_cmd + w[2]
        vd = a_cmd + w[3]

        return jnp.concatenate(
            (
                jnp.stack([xd, yd, hd, vd]),
                jnp.zeros(8, dtype=x.dtype),
            )
        )


def _zero_disturbance_input(t, x):
    del t, x
    return jnp.zeros(4, dtype=jnp.float32)


def _bounded_disturbance_input(t, x):
    del t
    x_int = irx.ut2i(x)
    d_lower = x_int.lower[8:12]
    d_upper = x_int.upper[8:12]
    d_center = 0.5 * (d_lower + d_upper)
    d_radius = 0.5 * (d_upper - d_lower)
    return irx.icentpert(d_center, d_radius)


_BRAKING_SYS = TurtleBotBraking()
_BRAKING_EMBSYS = irx.natemb(_BRAKING_SYS)


# ---------------------------------------------------------------------------
# Reach tube computation
# ---------------------------------------------------------------------------

def compute_reach_tube(w_des: float, v_des: float,
                       t_plan: float, t_stop: float,
                       z0, init_uncertainty: float = 0.01,
                       dt: float = 0.01,
                       disturbance_bound: float = 0.0,
                       disturbance_interval=None):
    """Compute the reachable tube for the turtlebot tracking a braking trajectory.

    Parameters
    ----------
    w_des, v_des      : planned yaw rate (rad/s) and speed (m/s)
    t_plan, t_stop    : planning horizon and braking duration (s)
    z0                : initial state [x, y, heading, v]
    init_uncertainty  : half-width ε_pos applied to position states (x,y).
                        Heading and speed uncertainties use ε_other = 0.2 * ε_pos.
    dt                : integration step size (s)
    disturbance_bound : half-width of bounded additive disturbance on each state
                        derivative component [dpx, dpy, dh, dv]
    disturbance_interval : optional signed disturbance interval [lower, upper]
                           for each additive disturbance component

    Returns
    -------
    embtraj  : irx Trajectory with ys shape (N, 8) — interval coordinates
    nom_traj : irx Trajectory with ys shape (N, 4) — nominal point trajectory
    """
    t0 = jnp.float32(0.0)
    tf = jnp.float32(t_plan + t_stop)

    z0_j = jnp.array(z0, dtype=jnp.float32)
    eps_pos = float(init_uncertainty)
    eps_other = 0.2 * eps_pos
    d_lo, d_hi = _normalize_disturbance_interval(disturbance_interval, disturbance_bound)
    d_center = 0.5 * (d_lo + d_hi)
    d_radius = 0.5 * (d_hi - d_lo)
    x0_j = jnp.concatenate(
        (
            z0_j,
            jnp.array([w_des, v_des, t_plan, t_stop], dtype=jnp.float32),
            jnp.array(d_center, dtype=jnp.float32),
        )
    )
    eps_vec = jnp.array(
        [eps_pos, eps_pos, eps_other, eps_other] + [0.0] * 4 + list(np.asarray(d_radius, dtype=float)),
        dtype=jnp.float32,
    )
    iz0 = irx.icentpert(x0_j, eps_vec)

    nom_traj = _BRAKING_SYS.compute_trajectory(
        t0, tf, x0_j, (_zero_disturbance_input,), dt, solver='rk45'
    )
    embtraj = _BRAKING_EMBSYS.compute_trajectory(
        t0, tf, irx.i2ut(iz0), (_bounded_disturbance_input,), dt, solver='rk45'
    )

    return embtraj, nom_traj


def warmup_verifier(dt: float = 0.03):
    """Trigger JIT compilation for the steady-state verification path."""
    compute_reach_tube(
        w_des=0.0,
        v_des=0.75,
        t_plan=0.5,
        t_stop=1.0,
        z0=[0.0, 0.0, 0.0, 0.75],
        init_uncertainty=0.01,
        dt=dt,
        disturbance_bound=np.zeros(4, dtype=float),
    )


# ---------------------------------------------------------------------------
# Obstacle collision check
# ---------------------------------------------------------------------------

def _normalize_polygon(poly):
    arr = np.asarray(poly, dtype=float)
    if arr.ndim != 2:
        raise ValueError('polygon must be a 2-D array')
    if arr.shape[0] != 2 and arr.shape[1] == 2:
        arr = arr.T
    if arr.shape[0] != 2:
        raise ValueError('polygon must have shape (2, N) or (N, 2)')
    if arr.shape[1] < 3:
        raise ValueError('polygon must have at least 3 vertices')
    if not np.allclose(arr[:, 0], arr[:, -1]):
        arr = np.hstack([arr, arr[:, :1]])
    return arr


def _rect_to_poly(rect):
    ox_lo, ox_hi, oy_lo, oy_hi = rect
    return np.array(
        [[ox_lo, ox_hi, ox_hi, ox_lo, ox_lo],
         [oy_lo, oy_lo, oy_hi, oy_hi, oy_lo]],
        dtype=float,
    )


def _poly_bounds(poly):
    poly_n = _normalize_polygon(poly)
    return (
        float(np.min(poly_n[0, :])),
        float(np.max(poly_n[0, :])),
        float(np.min(poly_n[1, :])),
        float(np.max(poly_n[1, :])),
    )


def _inflate_polygon(poly, radius):
    poly_n = _normalize_polygon(poly)
    radius = max(float(radius), 0.0)
    if radius <= 0.0:
        return poly_n
    if not _SHAPELY:
        x_lo, x_hi, y_lo, y_hi = _poly_bounds(poly_n)
        return _rect_to_poly((x_lo - radius, x_hi + radius, y_lo - radius, y_hi + radius))
    shp = ShapelyPolygon(poly_n.T)
    try:
        buffered = shp.buffer(radius, join_style='mitre')
    except TypeError:
        buffered = shp.buffer(radius, join_style=2)
    return np.array(buffered.exterior.coords, dtype=float).T


def _expand_obstacles(obs_rects=None, obstacle_polys=None, inflate_radius: float = 0.0):
    if obstacle_polys is not None and len(obstacle_polys) > 0:
        return [_inflate_polygon(poly, inflate_radius) for poly in obstacle_polys]
    obs_rects = [] if obs_rects is None else list(obs_rects)
    return [
        (ox_lo - inflate_radius, ox_hi + inflate_radius,
         oy_lo - inflate_radius, oy_hi + inflate_radius)
        for (ox_lo, ox_hi, oy_lo, oy_hi) in obs_rects
    ]


def _rect_overlap(a, b):
    ax_lo, ax_hi, ay_lo, ay_hi = a
    bx_lo, bx_hi, by_lo, by_hi = b
    return (ax_lo <= bx_hi and ax_hi >= bx_lo and
            ay_lo <= by_hi and ay_hi >= by_lo)


def check_obstacle_collision(embtraj, obs_rects, obstacle_inflate_radius: float = 0.0, obstacle_polys=None):
    """Check if the reachable tube (xy projection) intersects any obstacle.

    Parameters
    ----------
    embtraj      : embedded trajectory from compute_reach_tube
    obs_rects    : list of (x_lo, x_hi, y_lo, y_hi) world-frame obstacle bounds
    obstacle_inflate_radius : expand each obstacle bound by this radius (m)
    obstacle_polys : optional list of obstacle polygons in world frame

    Returns
    -------
    collision      : bool — True if a collision is detected
    collision_time : float or None — time of first detected collision
    xy_tube        : (N, 4) array of [x_lo, x_hi, y_lo, y_hi] per time step
    xy_swept       : (N-1, 4) array of interval hulls between consecutive steps
    expanded_obs   : list of expanded obstacle geometries
    ts_tube        : (N,) array of time stamps corresponding to xy_tube
    """
    ys = np.array(embtraj.ys)
    ts = np.array(embtraj.ts)

    finite  = np.isfinite(ts)
    ys, ts  = ys[finite], ts[finite]
    N       = ys.shape[0]

    n_state = ys.shape[1] // 2 if ys.ndim == 2 and ys.shape[1] > 0 else 0
    lower = ys[:, :n_state] if n_state > 0 else np.zeros((N, 0))
    upper = ys[:, n_state:] if n_state > 0 else np.zeros((N, 0))
    xy_tube = np.column_stack((lower[:, 0], upper[:, 0], lower[:, 1], upper[:, 1])) if n_state >= 2 else np.zeros((N, 4))
    xy_swept = np.zeros((max(N - 1, 0), 4))
    collision      = False
    collision_time = None
    collision_info = None
    expanded_obs = _expand_obstacles(obs_rects, obstacle_polys=obstacle_polys, inflate_radius=obstacle_inflate_radius)
    using_polys = obstacle_polys is not None and len(obstacle_polys) > 0
    if using_polys:
        obs_arr = np.asarray([_poly_bounds(poly) for poly in expanded_obs], dtype=float).reshape((-1, 4)) if len(expanded_obs) > 0 else np.zeros((0, 4))
        obs_geoms = [ShapelyPolygon(_normalize_polygon(poly).T) for poly in expanded_obs] if _SHAPELY else None
    else:
        obs_arr = np.asarray(expanded_obs, dtype=float).reshape((-1, 4)) if len(expanded_obs) > 0 else np.zeros((0, 4))
        obs_geoms = None

    if N > 0 and obs_arr.shape[0] > 0:
        sample_overlap = (
            (xy_tube[:, None, 0] <= obs_arr[None, :, 1]) &
            (xy_tube[:, None, 1] >= obs_arr[None, :, 0]) &
            (xy_tube[:, None, 2] <= obs_arr[None, :, 3]) &
            (xy_tube[:, None, 3] >= obs_arr[None, :, 2])
        )
        sample_hit_rows = np.any(sample_overlap, axis=1)
        if np.any(sample_hit_rows):
            k = int(np.argmax(sample_hit_rows))
            rect_geom = shapely_box(xy_tube[k][0], xy_tube[k][2], xy_tube[k][1], xy_tube[k][3]) if using_polys and _SHAPELY else None
            for obs_idx in np.flatnonzero(sample_overlap[k]):
                if rect_geom is not None and not rect_geom.intersects(obs_geoms[int(obs_idx)]):
                    continue
                collision = True
                collision_time = float(ts[k])
                collision_info = {
                    'kind': 'sample',
                    'k': int(k),
                    'tube_rect': tuple(xy_tube[k].tolist()),
                    'swept_rect': None,
                    'obs_idx': int(obs_idx),
                    'obs_rect': tuple(obs_arr[int(obs_idx)].tolist()),
                }
                break

    # Conservative inter-sample collision check via swept interval hulls.
    if N > 1:
        xy_swept[:, 0] = np.minimum(xy_tube[:-1, 0], xy_tube[1:, 0])
        xy_swept[:, 1] = np.maximum(xy_tube[:-1, 1], xy_tube[1:, 1])
        xy_swept[:, 2] = np.minimum(xy_tube[:-1, 2], xy_tube[1:, 2])
        xy_swept[:, 3] = np.maximum(xy_tube[:-1, 3], xy_tube[1:, 3])

        if not collision and obs_arr.shape[0] > 0:
            swept_overlap = (
                (xy_swept[:, None, 0] <= obs_arr[None, :, 1]) &
                (xy_swept[:, None, 1] >= obs_arr[None, :, 0]) &
                (xy_swept[:, None, 2] <= obs_arr[None, :, 3]) &
                (xy_swept[:, None, 3] >= obs_arr[None, :, 2])
            )
            swept_hit_rows = np.any(swept_overlap, axis=1)
            if np.any(swept_hit_rows):
                k = int(np.argmax(swept_hit_rows))
                swept_geom = shapely_box(xy_swept[k][0], xy_swept[k][2], xy_swept[k][1], xy_swept[k][3]) if using_polys and _SHAPELY else None
                for obs_idx in np.flatnonzero(swept_overlap[k]):
                    if swept_geom is not None and not swept_geom.intersects(obs_geoms[int(obs_idx)]):
                        continue
                    collision = True
                    collision_time = float(ts[k])
                    collision_info = {
                        'kind': 'swept',
                        'k': int(k),
                        'tube_rect': tuple(xy_tube[k].tolist()),
                        'tube_rect_next': tuple(xy_tube[k + 1].tolist()),
                        'swept_rect': tuple(xy_swept[k].tolist()),
                        'obs_idx': int(obs_idx),
                        'obs_rect': tuple(obs_arr[int(obs_idx)].tolist()),
                    }
                    break

    return collision, collision_time, xy_tube, xy_swept, expanded_obs, ts, collision_info


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def verify(w_des: float, v_des: float,
           t_plan: float, t_stop: float,
           z0, obstacle_rects,
           obstacle_polys=None,
           robot_radius: float = 0.0,
           obstacle_inflate_radius: float = 0.0,
           init_uncertainty: float = 0.01,
           dt: float = 0.01,
           disturbance_bound: float = 0.0,
           verbose: bool = True,
           disturbance_interval=None):
    """Run full immrax reachability verification for a Turtlebot braking trajectory.

    Parameters
    ----------
    w_des, v_des      : planned yaw rate (rad/s) and speed (m/s)
    t_plan, t_stop    : planning horizon and braking time (s)
    z0                : initial state [x, y, heading, v]
    obstacle_rects    : list of (x_lo, x_hi, y_lo, y_hi) — raw obstacle bounding boxes
    obstacle_polys    : optional list of obstacle polygons in world frame
    robot_radius      : robot footprint radius used as baseline positional uncertainty
    obstacle_inflate_radius : extra inflation radius for obstacle collision checks
    init_uncertainty  : additional position uncertainty radius ε_add (meters)
                        total ε_pos = robot_radius + ε_add
                        heading/speed uncertainty is ε_other = 0.2*ε_pos
    dt                : integration step (s)
    disturbance_bound : half-width of bounded additive disturbance on each
                        state derivative component
    disturbance_interval : optional signed disturbance interval [lower, upper]
                           for each additive disturbance component

    Returns
    -------
    dict with keys:
        'safe'           : bool — True if no collision detected in reach tube
        'collision_time' : float or None — time of first detected collision
        'xy_tube'        : (N, 4) array — [x_lo, x_hi, y_lo, y_hi] per step
        'ts_tube'        : (N,) array — times for xy_tube samples
        'nom_xy'         : (M, 2) array — nominal [x, y] trajectory
    """
    eps_add = max(float(init_uncertainty), 0.0)
    eps_pos = float(robot_radius) + eps_add
    eps_other = 0.2 * eps_pos
    d_lo, d_hi = _normalize_disturbance_interval(disturbance_interval, disturbance_bound)
    d_vec = np.maximum(np.abs(d_lo), np.abs(d_hi))
    if verbose:
        print(
            '  [immrax] reach tube  '
            f'(ε_pos=footprint+add=±({robot_radius:.3f}+{eps_add:.3f})=±{eps_pos:.3f}, '
            f'ε_heading/speed=±{eps_other:.3f}, dt={dt:.3f} s, '
            f'd_interval=[{d_lo[0]:.3f}, {d_hi[0]:.3f}]x, [{d_lo[1]:.3f}, {d_hi[1]:.3f}]y, '
            f'[{d_lo[2]:.3f}, {d_hi[2]:.3f}]h, [{d_lo[3]:.3f}, {d_hi[3]:.3f}]v, '
            f'obs_inflate=±{float(obstacle_inflate_radius):.3f})...'
        )
    embtraj, nom_traj = compute_reach_tube(
        w_des,
        v_des,
        t_plan,
        t_stop,
        z0,
        eps_pos,
        dt,
        disturbance_bound=disturbance_bound,
        disturbance_interval=disturbance_interval,
    )

    if verbose:
        print('  [immrax] checking obstacle intersections...')
    collision, t_coll, xy_tube, xy_swept, expanded_obs, ts_tube, collision_info = check_obstacle_collision(
        embtraj, obstacle_rects, obstacle_inflate_radius, obstacle_polys=obstacle_polys
    )

    # Extract nominal xy
    ys_nom = np.array(nom_traj.ys)
    ts_nom = np.array(nom_traj.ts)
    nom_xy = ys_nom[np.isfinite(ts_nom), :2]

    verdict = 'SAFE' if not collision else 'COLLISION DETECTED'
    suffix  = f'  (t = {t_coll:.3f} s)' if collision else ''
    if verbose:
        print(f'  [immrax] {verdict}{suffix}')

    return {
        'safe':           not collision,
        'collision_time': t_coll,
        'xy_tube':        xy_tube,
        'ts_tube':        ts_tube,
        'xy_swept':       xy_swept,
        'expanded_obs':   expanded_obs,
        'collision_info': collision_info,
        'positional_uncertainty_total': eps_pos,
        'positional_uncertainty_added': eps_add,
        'disturbance_bound': d_vec,
        'disturbance_interval': np.vstack([d_lo, d_hi]),
        'obstacle_inflate_radius': max(float(obstacle_inflate_radius), 0.0),
        'uncertainty_vec': np.array([eps_pos, eps_pos, eps_other, eps_other], dtype=float),
        'nom_xy':         nom_xy,
    }
