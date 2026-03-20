"""
Shared helpers for the disturbance-driven RTD case studies.

These utilities build random multi-gap courses, simulate actual disturbed
execution, run a standard-RTD baseline versus RTD-RAX, and produce summary
plots / animations.
"""

import os
import sys
import time
import copy
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from scipy.optimize import Bounds, minimize

sys.path.insert(0, os.path.dirname(__file__))

from frs_loader import _DEFAULT_DIR, k_to_wv, load_frs
from geometry_utils import FRS_to_world, compute_turtlebot_discretized_obs, compute_turtlebot_point_spacing, world_to_local
from trajectory import make_turtlebot_braking_trajectory
from turtlebot_agent import TurtlebotAgent
from polynomial_utils import (
    evaluate_frs_polynomial_on_obstacle_points,
    get_constraint_polynomial_gradient,
    get_frs_polynomial_structure,
)
from cost import turtlebot_cost_and_grad
from constraints import build_constraint


CASE_STUDY_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'Nimbus Roman No9 L', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.titlesize': 20,
    'axes.labelsize': 17,
    'axes.titleweight': 'medium',
    'axes.labelweight': 'medium',
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'figure.titlesize': 22,
    'figure.titleweight': 'medium',
}

DEFAULT_OBSTACLE_BUFFER = 0.05

_FRS_PATHS = {
    'standard': os.path.join(_DEFAULT_DIR, 'turtlebot_FRS_deg_10_v_0_0.5_to_1.0_preproc.mat'),
    'noerror': os.path.join(_DEFAULT_DIR, 'turtlebot_FRS_deg_10_v_0_0.5_to_1.0_noerror_preproc.mat'),
}


@dataclass
class DisturbancePatch:
    rect: tuple[float, float, float, float]
    disturbance: np.ndarray
    label: str = ''


@dataclass
class Scenario:
    name: str
    start_pose: np.ndarray
    goal: np.ndarray
    road_half_width: float | None
    world_bounds: tuple[float, float, float, float] | None
    obstacle_polys: list
    obstacle_rects: list
    patches: list
    seed: int


def apply_case_study_style():
    plt.rcParams.update(CASE_STUDY_STYLE)


def ensure_parent_dir(path):
    if path is None:
        return
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _coerce_disturbance_bound(value, size=4, name='disturbance bound'):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(int(size), float(arr), dtype=float)
    arr = arr.reshape(-1)
    if arr.size == 1:
        return np.full(int(size), float(arr[0]), dtype=float)
    if arr.size != int(size):
        raise ValueError(f'{name} must be a scalar or length-{int(size)} sequence')
    return arr.astype(float, copy=True)


def make_rect_polygon(x_lo, x_hi, y_lo, y_hi):
    return np.array(
        [[x_lo, x_hi, x_hi, x_lo, x_lo],
         [y_lo, y_lo, y_hi, y_hi, y_lo]],
        dtype=float,
    )


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


def polygon_bounds(poly):
    poly_n = _normalize_polygon(poly)
    return (
        float(np.min(poly_n[0, :])),
        float(np.max(poly_n[0, :])),
        float(np.min(poly_n[1, :])),
        float(np.max(poly_n[1, :])),
    )


def _make_random_polygon_rng(rng, center, scale_x, scale_y, n_vertices_range=(5, 8), rotation=None):
    n_min, n_max = n_vertices_range
    n_vertices = int(rng.integers(int(n_min), int(n_max) + 1))
    angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=n_vertices))
    radii = rng.uniform(0.65, 1.00, size=n_vertices)
    pts = np.vstack(
        [
            float(scale_x) * radii * np.cos(angles),
            float(scale_y) * radii * np.sin(angles),
        ]
    )
    theta = float(rng.uniform(-np.pi, np.pi) if rotation is None else rotation)
    c = np.cos(theta)
    s = np.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    pts = rot @ pts + np.asarray(center, dtype=float).reshape(2, 1)
    return _normalize_polygon(pts)


def _clip_polygon_to_bounds(poly, world_bounds, margin=0.18):
    x_lo_b, x_hi_b, y_lo_b, y_hi_b = world_bounds
    poly_n = _normalize_polygon(poly).copy()
    x_lo, x_hi, y_lo, y_hi = polygon_bounds(poly_n)
    dx = 0.0
    dy = 0.0
    if x_lo < x_lo_b + margin:
        dx = (x_lo_b + margin) - x_lo
    elif x_hi > x_hi_b - margin:
        dx = (x_hi_b - margin) - x_hi
    if y_lo < y_lo_b + margin:
        dy = (y_lo_b + margin) - y_lo
    elif y_hi > y_hi_b - margin:
        dy = (y_hi_b - margin) - y_hi
    poly_n[0, :] += dx
    poly_n[1, :] += dy
    return poly_n


def _edge_distance_sq(point, a, b):
    p = np.asarray(point, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        d = p - a
        return float(np.dot(d, d))
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    proj = a + t * ab
    d = p - proj
    return float(np.dot(d, d))


def _point_in_polygon(point, poly):
    poly_n = _normalize_polygon(poly)
    x = float(point[0])
    y = float(point[1])
    inside = False
    for idx in range(poly_n.shape[1] - 1):
        x1 = float(poly_n[0, idx])
        y1 = float(poly_n[1, idx])
        x2 = float(poly_n[0, idx + 1])
        y2 = float(poly_n[1, idx + 1])
        if _edge_distance_sq((x, y), (x1, y1), (x2, y2)) <= 1e-16:
            return True
        intersects = ((y1 > y) != (y2 > y))
        if intersects:
            denom = y2 - y1
            if abs(denom) <= 1e-12:
                x_cross = x1
            else:
                x_cross = x1 + (y - y1) * (x2 - x1) / denom
            if x <= x_cross:
                inside = not inside
    return inside


def _circle_intersects_polygon(center, radius, poly):
    poly_n = _normalize_polygon(poly)
    center = np.asarray(center, dtype=float).reshape(2)
    r_sq = float(radius) * float(radius)
    if _point_in_polygon(center, poly_n):
        return True
    for idx in range(poly_n.shape[1] - 1):
        a = poly_n[:, idx]
        b = poly_n[:, idx + 1]
        if _edge_distance_sq(center, a, b) <= r_sq:
            return True
    return False


def _scenario_plot_limits(scenario, pad=0.5):
    if scenario.world_bounds is not None:
        x_lo, x_hi, y_lo, y_hi = scenario.world_bounds
        return (float(x_lo) - pad, float(x_hi) + pad), (float(y_lo) - pad, float(y_hi) + pad)

    xs = [float(scenario.start_pose[0]), float(scenario.goal[0, 0])]
    ys = [float(scenario.start_pose[1]), float(scenario.goal[1, 0])]
    if scenario.road_half_width is not None:
        ys.extend([float(-scenario.road_half_width), float(scenario.road_half_width)])
    for poly in scenario.obstacle_polys:
        poly_n = _normalize_polygon(poly)
        xs.extend(poly_n[0, :].tolist())
        ys.extend(poly_n[1, :].tolist())
    for patch in scenario.patches:
        x_lo, x_hi, y_lo, y_hi = patch.rect
        xs.extend([x_lo, x_hi])
        ys.extend([y_lo, y_hi])
    return (min(xs) - pad, max(xs) + pad), (min(ys) - pad, max(ys) + pad)


def _clip_gap_center(center, road_half_width, gap_width, margin=0.18):
    lim = max(0.0, road_half_width - gap_width / 2.0 - margin)
    return float(np.clip(center, -lim, lim))


def generate_gap_patch_course(
    seed,
    course_length=6.0,
    road_half_width=1.35,
    stage_count=3,
    stage_width_range=(0.38, 0.55),
    gap_width_range=(0.82, 1.00),
    gap_center_step_range=(-0.34, 0.34),
    patch_length_range=(0.55, 0.85),
    patch_gap_range=(0.10, 0.18),
    disturbance_y_range=(0.05, 0.11),
    disturbance_h_range=(0.04, 0.10),
    disturbance_v_range=(0.02, 0.06),
    title='Random Disturbance Course',
):
    rng = np.random.default_rng(seed)
    xs = np.linspace(1.35, course_length - 1.1, stage_count)
    xs = xs + rng.uniform(-0.18, 0.18, size=stage_count)

    obstacle_polys = []
    obstacle_rects = []
    patches = []

    gap_center = 0.0
    for idx, x_mid in enumerate(xs):
        gap_center = _clip_gap_center(
            gap_center + rng.uniform(*gap_center_step_range),
            road_half_width,
            rng.uniform(*gap_width_range),
        )
        gap_width = float(rng.uniform(*gap_width_range))
        gap_center = _clip_gap_center(gap_center, road_half_width, gap_width)
        stage_width = float(rng.uniform(*stage_width_range))
        x_lo = float(x_mid - stage_width / 2.0)
        x_hi = float(x_mid + stage_width / 2.0)
        gap_y_lo = float(gap_center - gap_width / 2.0)
        gap_y_hi = float(gap_center + gap_width / 2.0)

        if gap_y_hi < road_half_width - 0.08:
            rect = (x_lo, x_hi, gap_y_hi, road_half_width)
            obstacle_rects.append(rect)
            obstacle_polys.append(make_rect_polygon(*rect))
        if gap_y_lo > -road_half_width + 0.08:
            rect = (x_lo, x_hi, -road_half_width, gap_y_lo)
            obstacle_rects.append(rect)
            obstacle_polys.append(make_rect_polygon(*rect))

        push_sign = -1.0 if gap_center >= 0.0 else 1.0
        if abs(gap_center) < 0.1:
            push_sign = -1.0 if rng.random() < 0.5 else 1.0
        dpy = float(push_sign * rng.uniform(*disturbance_y_range))
        dh = float(push_sign * rng.uniform(*disturbance_h_range))
        dv = float(-rng.uniform(*disturbance_v_range))
        patch_len = float(rng.uniform(*patch_length_range))
        patch_gap = float(rng.uniform(*patch_gap_range))
        patch_rect = (
            float(x_lo - patch_gap - patch_len),
            float(x_lo - patch_gap),
            float(-road_half_width),
            float(road_half_width),
        )
        patches.append(
            DisturbancePatch(
                rect=patch_rect,
                disturbance=np.array([0.0, dpy, dh, dv], dtype=float),
                label=f'p{idx + 1}',
            )
        )

    return Scenario(
        name=title,
        start_pose=np.array([0.0, 0.0, 0.0], dtype=float),
        goal=np.array([[course_length], [0.0]], dtype=float),
        road_half_width=float(road_half_width),
        world_bounds=None,
        obstacle_polys=obstacle_polys,
        obstacle_rects=obstacle_rects,
        patches=patches,
        seed=int(seed),
    )


def generate_random_polygon_ice_world(
    seed,
    world_length=16.0,
    world_half_height=3.8,
    stage_count=7,
    rectangular_obstacles=False,
    passage_width_range=(1.05, 1.35),
    obstacle_scale_x_range=(0.40, 0.78),
    obstacle_scale_y_range=(0.62, 1.08),
    corridor_step_range=(-0.55, 0.55),
    obstacle_x_jitter=0.30,
    patch_length_range=(0.90, 1.55),
    patch_half_height_range=(1.30, 2.40),
    patch_gap_range=(0.10, 0.28),
    disturbance_y_range=(0.07, 0.12),
    disturbance_h_range=(0.05, 0.09),
    disturbance_v_range=(0.02, 0.05),
    title='Full Driving Scenario: Ice Disturbances Through Random Obstacles',
):
    rng = np.random.default_rng(seed)
    world_bounds = (0.0, float(world_length), -float(world_half_height), float(world_half_height))
    start_xy = np.array([1.0, -0.62 * float(world_half_height)], dtype=float)
    goal_xy = np.array([float(world_length) - 1.0, 0.58 * float(world_half_height)], dtype=float)
    xs = np.linspace(start_xy[0] + 1.6, goal_xy[0] - 1.6, int(stage_count))
    xs = xs + rng.uniform(-0.22, 0.22, size=xs.shape[0])

    corridor_base = np.linspace(start_xy[1], goal_xy[1], xs.shape[0])
    y_margin = 1.15
    corridor_y = []
    wander = 0.0
    for base in corridor_base:
        wander = np.clip(
            wander + rng.uniform(*corridor_step_range),
            -(float(world_half_height) - y_margin),
            +(float(world_half_height) - y_margin),
        )
        corridor_y.append(float(np.clip(base + wander, -float(world_half_height) + y_margin, float(world_half_height) - y_margin)))
    corridor_y = np.asarray(corridor_y, dtype=float)

    start_heading = float(np.arctan2(corridor_y[0] - start_xy[1], xs[0] - start_xy[0]))
    obstacle_polys = []
    obstacle_rects = []
    patches = []

    for idx, (x_mid, y_mid) in enumerate(zip(xs, corridor_y)):
        passage_width = float(rng.uniform(*passage_width_range))
        side = -1.0 if rng.random() < 0.5 else 1.0
        obs_center = np.array(
            [
                float(x_mid + rng.uniform(-obstacle_x_jitter, obstacle_x_jitter)),
                float(np.clip(y_mid + side * rng.uniform(0.55, 0.95), -float(world_half_height) + 0.8, float(world_half_height) - 0.8)),
            ],
            dtype=float,
        )
        sx = float(rng.uniform(*obstacle_scale_x_range))
        sy = float(rng.uniform(*obstacle_scale_y_range))
        if rectangular_obstacles:
            half_x = sx
            half_y = sy
            obs_bounds = (
                obs_center[0] - half_x, obs_center[0] + half_x,
                obs_center[1] - half_y, obs_center[1] + half_y,
            )
            target_edge = y_mid + side * (0.18 + 0.25 * max(passage_width - passage_width_range[0], 0.0))
            if side > 0.0:
                shift = target_edge - obs_bounds[2]
            else:
                shift = target_edge - obs_bounds[3]
            obs_bounds = (obs_bounds[0], obs_bounds[1], obs_bounds[2] + shift, obs_bounds[3] + shift)
            obs_bounds = (
                max(obs_bounds[0], world_bounds[0] + 0.18),
                min(obs_bounds[1], world_bounds[1] - 0.18),
                max(obs_bounds[2], world_bounds[2] + 0.18),
                min(obs_bounds[3], world_bounds[3] - 0.18),
            )
            obs_poly = make_rect_polygon(*obs_bounds)
        else:
            obs_poly = _make_random_polygon_rng(
                rng,
                obs_center,
                scale_x=sx,
                scale_y=sy,
            )
            obs_poly = _clip_polygon_to_bounds(obs_poly, world_bounds)
            obs_bounds = polygon_bounds(obs_poly)

            target_edge = y_mid + side * (0.18 + 0.25 * max(passage_width - passage_width_range[0], 0.0))
            if side > 0.0:
                obs_poly[1, :] += (target_edge - obs_bounds[2])
            else:
                obs_poly[1, :] += (target_edge - obs_bounds[3])
            obs_poly = _clip_polygon_to_bounds(obs_poly, world_bounds)
            obs_bounds = polygon_bounds(obs_poly)

        obstacle_polys.append(obs_poly)
        obstacle_rects.append(obs_bounds)

        push_sign = side
        patch_len = float(rng.uniform(*patch_length_range))
        patch_gap = float(rng.uniform(*patch_gap_range))
        patch_half = float(rng.uniform(*patch_half_height_range))
        patch_cy = float(np.clip(y_mid + rng.uniform(-0.12, 0.12), -float(world_half_height) + patch_half + 0.05, float(world_half_height) - patch_half - 0.05))
        patch_rect = (
            float(x_mid - patch_gap - patch_len),
            float(x_mid - patch_gap),
            float(patch_cy - patch_half),
            float(patch_cy + patch_half),
        )
        patches.append(
            DisturbancePatch(
                rect=patch_rect,
                disturbance=np.array(
                    [
                        float(push_sign * rng.uniform(*disturbance_y_range) * 0.25),
                        float(push_sign * rng.uniform(*disturbance_y_range)),
                        0.0,
                        0.0,
                    ],
                    dtype=float,
                ),
                label=f'ice{idx + 1}',
            )
        )

    stray_count = max(2, int(stage_count) // 3)
    ref_x = np.concatenate([[start_xy[0]], xs, [goal_xy[0]]])
    ref_y = np.concatenate([[start_xy[1]], corridor_y, [goal_xy[1]]])
    for sidx in range(stray_count):
        x_mid = float(rng.uniform(start_xy[0] + 1.2, goal_xy[0] - 1.0))
        y_ref = float(np.interp(x_mid, ref_x, ref_y))
        side = -1.0 if rng.random() < 0.5 else 1.0
        center = np.array(
            [
                x_mid,
                float(np.clip(y_ref + side * rng.uniform(1.4, 2.2), -float(world_half_height) + 0.8, float(world_half_height) - 0.8)),
            ],
            dtype=float,
        )
        stray_sx = float(rng.uniform(0.30, 0.58))
        stray_sy = float(rng.uniform(0.45, 0.82))
        if rectangular_obstacles:
            stray_bounds = (
                center[0] - stray_sx, center[0] + stray_sx,
                center[1] - stray_sy, center[1] + stray_sy,
            )
            stray_bounds = (
                max(stray_bounds[0], world_bounds[0] + 0.18),
                min(stray_bounds[1], world_bounds[1] - 0.18),
                max(stray_bounds[2], world_bounds[2] + 0.18),
                min(stray_bounds[3], world_bounds[3] - 0.18),
            )
            stray_poly = make_rect_polygon(*stray_bounds)
        else:
            stray_poly = _make_random_polygon_rng(
                rng,
                center,
                scale_x=stray_sx,
                scale_y=stray_sy,
            )
            stray_poly = _clip_polygon_to_bounds(stray_poly, world_bounds)
        obstacle_polys.append(stray_poly)
        obstacle_rects.append(polygon_bounds(stray_poly))

    return Scenario(
        name=title,
        start_pose=np.array([start_xy[0], start_xy[1], start_heading], dtype=float),
        goal=goal_xy.reshape(2, 1),
        road_half_width=None,
        world_bounds=world_bounds,
        obstacle_polys=obstacle_polys,
        obstacle_rects=obstacle_rects,
        patches=patches,
        seed=int(seed),
    )


def inset_road_edge_obstacles(scenario, inset):
    inset = float(inset)
    if inset <= 0.0:
        return scenario

    scenario_new = copy.deepcopy(scenario)
    obstacle_rects = []
    obstacle_polys = []
    for rect in scenario_new.obstacle_rects:
        x_lo, x_hi, y_lo, y_hi = rect
        if np.isclose(y_hi, scenario_new.road_half_width):
            y_hi = y_hi - inset
            if y_hi <= y_lo + 0.15:
                continue
        elif np.isclose(y_lo, -scenario_new.road_half_width):
            y_lo = y_lo + inset
            if y_hi <= y_lo + 0.15:
                continue
        rect_new = (float(x_lo), float(x_hi), float(y_lo), float(y_hi))
        obstacle_rects.append(rect_new)
        obstacle_polys.append(make_rect_polygon(*rect_new))

    scenario_new.obstacle_rects = obstacle_rects
    scenario_new.obstacle_polys = obstacle_polys
    return scenario_new


def load_case_study_models():
    models = {}
    for key, path in _FRS_PATHS.items():
        frs = load_frs(path=path)
        fp = get_frs_polynomial_structure(frs['pows'], frs['coef'], frs['z_cols'], frs['k_cols'])
        models[key] = (frs, fp)
    return models


def compute_frs_contour_world(frs, k_opt, initial_pose, grid_res=100):
    """Compute the FRS polynomial zero-level contour at k_opt in world coordinates.

    Returns a (2, M) array of world-frame contour points (with NaN separators
    between segments), or None if no contour is found.
    """
    z1g, z2g = np.meshgrid(np.linspace(-1, 1, grid_res), np.linspace(-1, 1, grid_res))
    z_grid = np.vstack([z1g.ravel(), z2g.ravel()])

    k_pows = frs['pows'][:, frs['k_cols']]
    z_pows = frs['pows'][:, frs['z_cols']]
    k_mono = np.prod(k_opt[np.newaxis, :] ** k_pows, axis=1)
    z_vals = np.prod(z_grid[np.newaxis, :, :] ** z_pows[:, :, np.newaxis], axis=1)
    frs_grid = (frs['coef'] * k_mono) @ z_vals

    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(z1g, z2g, frs_grid.reshape(z1g.shape), levels=[0.0])
    plt.close(fig_tmp)

    if not cs.allsegs or not cs.allsegs[0]:
        return None

    close_tol = 5.0 / float(grid_res)
    segs = [s for s in cs.allsegs[0]
            if s.size > 0 and np.linalg.norm(s[0] - s[-1]) <= close_tol]
    if not segs:
        segs = cs.allsegs[0]

    parts = []
    for i, seg in enumerate(segs):
        parts.append(seg.T)
        if i < len(segs) - 1:
            parts.append(np.full((2, 1), np.nan))
    C_FRS = np.hstack(parts) if parts else None
    if C_FRS is None:
        return None

    return FRS_to_world(C_FRS, initial_pose, frs['initial_x'], frs['initial_y'], frs['distance_scale'])


def compute_step_contours(result, frs, grid_res=100):
    """Compute world-frame FRS contours for each step in an episode result.

    Returns a list (one per step) of (2, M) arrays or None for infeasible steps.
    """
    contours = []
    for record in result['step_records']:
        if not record['feasible'] or 'k' not in record:
            contours.append(None)
            continue
        seg_start = record.get('segment_start', 0)
        pose = result['agent'].state[:, seg_start]
        contours.append(compute_frs_contour_world(frs, record['k'], pose, grid_res=grid_res))
    return contours


def compute_path_arclength(result):
    st = np.asarray(result['agent'].state[:2, :], dtype=float)
    if st.shape[1] < 2:
        return 0.0
    delta = np.diff(st, axis=1)
    return float(np.sum(np.linalg.norm(delta, axis=0)))


def _precompute_obs_display(scenario, frs, initial_pose, footprint, obstacle_buffer):
    spacing = compute_turtlebot_point_spacing(footprint, obstacle_buffer)
    all_buf = []
    all_pts = []
    for poly in scenario.obstacle_polys:
        _, obs_buf, obs_pts = compute_turtlebot_discretized_obs(
            poly,
            initial_pose,
            obstacle_buffer,
            spacing,
            frs,
        )
        all_buf.append(obs_buf)
        all_pts.append(obs_pts)
    return all_buf, all_pts


def disturbance_from_patches(state, patches):
    x = float(state[0])
    y = float(state[1])
    out = np.zeros(4, dtype=float)
    for patch in patches:
        x_lo, x_hi, y_lo, y_hi = patch.rect
        if x_lo <= x <= x_hi and y_lo <= y <= y_hi:
            out += patch.disturbance
    return out


def disturbance_interval_from_world_traj(z_world, patches):
    if z_world.shape[1] == 0 or len(patches) == 0:
        zeros = np.zeros(4, dtype=float)
        return zeros, zeros
    vals = np.array([disturbance_from_patches(z_world[:, i], patches) for i in range(z_world.shape[1])], dtype=float)
    if vals.size == 0:
        zeros = np.zeros(4, dtype=float)
        return zeros, zeros
    return np.min(vals, axis=0), np.max(vals, axis=0)


def disturbance_bound_from_world_traj(z_world, patches):
    d_lo, d_hi = disturbance_interval_from_world_traj(z_world, patches)
    return np.maximum(np.abs(d_lo), np.abs(d_hi))


def worst_case_disturbance_bound(z_world, patches):
    """Worst-case disturbance from ALL patches whose x-range overlaps the trajectory.

    Unlike ``disturbance_bound_from_world_traj`` which only samples disturbance at
    the exact (x,y) positions along the planned path, this computes the envelope
    over every patch the robot *could* drift into laterally.  RTD-RAX uses this so
    that even if the robot is pushed sideways into an adjacent patch, the
    verification bound still covers it.
    """
    if z_world.shape[1] == 0 or len(patches) == 0:
        return np.zeros(4, dtype=float)
    traj_x_lo = float(z_world[0, :].min())
    traj_x_hi = float(z_world[0, :].max())
    d_bound = np.zeros(4, dtype=float)
    for patch in patches:
        px_lo, px_hi, _py_lo, _py_hi = patch.rect
        if px_hi >= traj_x_lo and px_lo <= traj_x_hi:
            d_bound = np.maximum(d_bound, np.abs(patch.disturbance))
    return d_bound


def corridor_disturbance_interval(z_world, patches, corridor_radius=0.175):
    """Signed disturbance interval from patches overlapping a trajectory corridor.

    For each trajectory point, expands an (x, y) box of half-width
    ``corridor_radius`` and includes any patch whose rectangle overlaps that box.
    This catches patches the robot could drift into (unlike trajectory-sampled,
    which only checks exact positions) while excluding y-distant patches that
    cannot be reached (unlike worst-case, which only checks x-range overlap).

    Returns ``(d_lo, d_hi)`` — a signed interval, preserving directionality so
    the reach tube only grows in the direction the disturbance actually pushes.
    """
    if z_world.shape[1] == 0 or len(patches) == 0:
        zeros = np.zeros(4, dtype=float)
        return zeros, zeros
    corridor_radius = max(float(corridor_radius), 0.0)
    traj_x = z_world[0, :]
    traj_y = z_world[1, :]
    d_lo = np.zeros(4, dtype=float)
    d_hi = np.zeros(4, dtype=float)
    for patch in patches:
        px_lo, px_hi, py_lo, py_hi = patch.rect
        x_overlap = (traj_x + corridor_radius >= px_lo) & (traj_x - corridor_radius <= px_hi)
        y_overlap = (traj_y + corridor_radius >= py_lo) & (traj_y - corridor_radius <= py_hi)
        if np.any(x_overlap & y_overlap):
            d_lo = np.minimum(d_lo, patch.disturbance)
            d_hi = np.maximum(d_hi, patch.disturbance)
    return d_lo, d_hi


def transform_reference_to_world(initial_pose, z_ref):
    c = np.cos(float(initial_pose[2]))
    s = np.sin(float(initial_pose[2]))
    rot = np.array([[c, -s], [s, c]], dtype=float)
    xy_world = rot @ np.asarray(z_ref[:2, :], dtype=float) + np.asarray(initial_pose[:2], dtype=float).reshape(2, 1)
    h_world = np.asarray(z_ref[2, :], dtype=float) + float(initial_pose[2])
    return np.vstack([xy_world, h_world, np.asarray(z_ref[3, :], dtype=float)])


def truncate_reference(t_ref, u_ref, z_ref, t_exec):
    t_exec = float(np.clip(t_exec, 0.0, float(t_ref[-1])))
    if t_exec >= float(t_ref[-1]) - 1e-12:
        return t_ref, u_ref, z_ref
    keep = t_ref < t_exec
    t_new = np.concatenate([t_ref[keep], np.array([t_exec])])
    u_new = np.vstack([np.interp(t_new, t_ref, u_ref[i, :]) for i in range(u_ref.shape[0])])
    z_new = np.vstack([np.interp(t_new, t_ref, z_ref[i, :]) for i in range(z_ref.shape[0])])
    return t_new, u_new, z_new


def _solve_step(frs, fp, state, speed, goal_world, spacing, obstacle_polys, k_init, obstacle_buffer):
    solve_start = time.perf_counter()
    z_goal_local = np.asarray(world_to_local(state, goal_world)).reshape(-1)
    x_des_loc = float(z_goal_local[0])
    y_des_loc = float(z_goal_local[1])

    o_frs_parts = []
    for poly in obstacle_polys:
        o_frs_i, _, _ = compute_turtlebot_discretized_obs(poly, state, obstacle_buffer, spacing, frs)
        if o_frs_i.shape[1] > 0:
            o_frs_parts.append(o_frs_i)
    o_frs = np.hstack(o_frs_parts) if o_frs_parts else np.zeros((2, 0))

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

    optimize_start = time.perf_counter()
    res = minimize(
        fun=lambda k: turtlebot_cost_and_grad(k, frs['w_max'], v_max, x_des_loc, y_des_loc),
        x0=np.asarray(k_init, dtype=float),
        jac=True,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 220, 'ftol': 1e-6, 'disp': False},
    )
    optimize_time = time.perf_counter() - optimize_start
    feasible = bool(res.success or res.status == 0)
    timing = {
        'solve_setup_time': float(optimize_start - solve_start),
        'solve_optimize_time': float(optimize_time),
        'solve_time': float(time.perf_counter() - solve_start),
    }
    return feasible, (res.x if feasible else None), res, timing


def _prepare_candidate(frs, agent, k_vec, patches, corridor_radius=0.0):
    w_des, v_des = k_to_wv(k_vec, frs)
    t_plan = frs['t_plan']
    t_stop = max(v_des / agent.max_accel, 1e-3)
    t_ref, u_ref, z_ref = make_turtlebot_braking_trajectory(t_plan, t_stop, w_des, v_des)
    z_world = transform_reference_to_world(agent.state[:, -1], z_ref)
    d_lo, d_hi = disturbance_interval_from_world_traj(z_world, patches)
    d_bound = np.maximum(np.abs(d_lo), np.abs(d_hi))
    wc_bound = worst_case_disturbance_bound(z_world, patches)
    corr_d_lo, corr_d_hi = corridor_disturbance_interval(z_world, patches, corridor_radius)
    return {
        'k': np.asarray(k_vec, dtype=float),
        'w_des': float(w_des),
        'v_des': float(v_des),
        't_plan': float(t_plan),
        't_stop': float(t_stop),
        't_ref': t_ref,
        'u_ref': u_ref,
        'z_ref': z_ref,
        'z_world': z_world,
        'disturbance_bound': d_bound,
        'worst_case_disturbance_bound': wc_bound,
        'disturbance_interval': np.vstack([d_lo, d_hi]),
        'corridor_disturbance_interval': np.vstack([corr_d_lo, corr_d_hi]),
    }


def _candidate_for_verification(candidate, agent, patches, verify_horizon, corridor_radius=0.0):
    if verify_horizon is None:
        return candidate
    verify_horizon = float(verify_horizon)
    if verify_horizon >= float(candidate['t_ref'][-1]) - 1e-12:
        return candidate
    t_ref_v, u_ref_v, z_ref_v = truncate_reference(
        candidate['t_ref'],
        candidate['u_ref'],
        candidate['z_ref'],
        verify_horizon,
    )
    z_world_v = transform_reference_to_world(agent.state[:, -1], z_ref_v)
    d_lo, d_hi = disturbance_interval_from_world_traj(z_world_v, patches)
    wc_bound = worst_case_disturbance_bound(z_world_v, patches)
    corr_d_lo, corr_d_hi = corridor_disturbance_interval(z_world_v, patches, corridor_radius)
    out = dict(candidate)
    out.update(
        {
            't_plan': float(t_ref_v[-1]),
            't_stop': 0.0,
            't_ref': t_ref_v,
            'u_ref': u_ref_v,
            'z_ref': z_ref_v,
            'z_world': z_world_v,
            'disturbance_bound': np.maximum(np.abs(d_lo), np.abs(d_hi)),
            'worst_case_disturbance_bound': wc_bound,
            'disturbance_interval': np.vstack([d_lo, d_hi]),
            'corridor_disturbance_interval': np.vstack([corr_d_lo, corr_d_hi]),
        }
    )
    return out


def _verify_candidate(candidate, agent, scenario, verify_uncertainty, verify_dt, obstacle_polys=None, use_worst_case_disturbance=False, obstacle_inflate_radius=0.0, corridor_radius=0.0, verify_disturbance=0.0):
    from immrax_verify import verify as immrax_verify

    d_interval = None
    if corridor_radius > 0.0:
        d_interval = candidate.get('corridor_disturbance_interval')

    if d_interval is not None:
        d_bound = np.zeros(4, dtype=float)
    elif use_worst_case_disturbance:
        d_bound = candidate.get('worst_case_disturbance_bound', candidate['disturbance_bound'])
    else:
        d_bound = candidate['disturbance_bound']
    d_bound = _coerce_disturbance_bound(d_bound, name='verification disturbance bound')

    verify_floor = _coerce_disturbance_bound(verify_disturbance, name='verify_disturbance')
    if np.any(verify_floor > 0.0):
        d_bound = np.maximum(d_bound, verify_floor)

    return immrax_verify(
        w_des=candidate['w_des'],
        v_des=candidate['v_des'],
        t_plan=candidate['t_plan'],
        t_stop=candidate['t_stop'],
        z0=agent.state[:, -1].tolist(),
        obstacle_rects=scenario.obstacle_rects,
        obstacle_polys=obstacle_polys,
        robot_radius=agent.footprint,
        obstacle_inflate_radius=float(obstacle_inflate_radius),
        init_uncertainty=verify_uncertainty,
        dt=verify_dt,
        disturbance_bound=d_bound,
        disturbance_interval=d_interval,
        verbose=False,
    )


def _timed_prepare_candidate(frs, agent, k_vec, patches, corridor_radius=0.0):
    start = time.perf_counter()
    candidate = _prepare_candidate(frs, agent, k_vec, patches, corridor_radius=corridor_radius)
    return candidate, float(time.perf_counter() - start)


def _timed_verify_candidate(
    candidate,
    agent,
    patches,
    verify_horizon,
    scenario,
    verify_uncertainty,
    verify_dt,
    obstacle_polys=None,
    use_worst_case_disturbance=False,
    corridor_radius=0.0,
    obstacle_inflate_radius=0.0,
    verify_disturbance=0.0,
):
    start = time.perf_counter()
    verify_candidate = _candidate_for_verification(candidate, agent, patches, verify_horizon, corridor_radius=corridor_radius)
    result = _verify_candidate(
        verify_candidate,
        agent,
        scenario,
        verify_uncertainty,
        verify_dt,
        obstacle_polys=obstacle_polys,
        use_worst_case_disturbance=use_worst_case_disturbance,
        corridor_radius=corridor_radius,
        obstacle_inflate_radius=obstacle_inflate_radius,
        verify_disturbance=verify_disturbance,
    )
    return result, float(time.perf_counter() - start)


def _first_collision(segment, scenario, robot_radius, start_idx=1):
    start_idx = max(int(start_idx), 0)
    for idx in range(start_idx, segment.shape[1]):
        x = float(segment[0, idx])
        y = float(segment[1, idx])
        if scenario.world_bounds is not None:
            x_lo, x_hi, y_lo, y_hi = scenario.world_bounds
            if (
                x <= x_lo + robot_radius
                or x >= x_hi - robot_radius
                or y <= y_lo + robot_radius
                or y >= y_hi - robot_radius
            ):
                return idx, {'kind': 'world_edge', 'point': (x, y), 'bounds': tuple(scenario.world_bounds)}
        elif scenario.road_half_width is not None and abs(y) >= scenario.road_half_width - robot_radius:
            return idx, {'kind': 'road_edge', 'point': (x, y)}

        for obs_idx, poly in enumerate(scenario.obstacle_polys):
            rect = scenario.obstacle_rects[obs_idx]
            if (
                x < rect[0] - robot_radius
                or x > rect[1] + robot_radius
                or y < rect[2] - robot_radius
                or y > rect[3] + robot_radius
            ):
                continue
            if _circle_intersects_polygon((x, y), robot_radius, poly):
                return idx, {
                    'kind': 'obstacle',
                    'point': (x, y),
                    'obs_idx': int(obs_idx),
                    'bounds': tuple(rect),
                }
    return None, None


def _first_goal_hit(segment, goal_world, goal_tol, start_idx=1):
    start_idx = max(int(start_idx), 0)
    goal_xy = np.asarray(goal_world, dtype=float).reshape(2)
    for idx in range(start_idx, segment.shape[1]):
        if np.linalg.norm(segment[:2, idx] - goal_xy) <= goal_tol:
            return idx
    return None


def _execute_emergency_brake(agent, scenario, speed):
    if speed <= 1e-6:
        return {'collision': False, 'collision_info': None, 'segment_start': agent.state.shape[1] - 1}
    t_stop = max(speed / agent.max_accel, 1e-3)
    t_ref, u_ref, z_ref = make_turtlebot_braking_trajectory(0.0, t_stop, 0.0, speed)
    seg_start = agent.state.shape[1] - 1
    agent.move(
        t_ref[-1],
        t_ref,
        u_ref,
        z_ref,
        disturbance=lambda t, z: disturbance_from_patches(z, scenario.patches),
    )
    segment = agent.state[:, seg_start:]
    hit_idx, collision = _first_collision(segment, scenario, agent.footprint)
    if hit_idx is not None:
        gidx = seg_start + int(hit_idx)
        agent.state = agent.state[:, :gidx + 1]
        agent.time = agent.time[:gidx + 1]
    return {'collision': hit_idx is not None, 'collision_info': collision, 'segment_start': seg_start}


def run_episode(
    scenario,
    planner,
    models,
    v0=0.75,
    max_steps=60,
    goal_tol=0.12,
    speed_tol=0.08,
    t_move=0.45,
    obstacle_buffer=DEFAULT_OBSTACLE_BUFFER,
    verify_uncertainty=0.025,
    verify_dt=0.01,
    verify_horizon=None,
    repair_max_iters=4,
    repair_speed_backoff=0.08,
    repair_buffer_step=0.015,
    repair_push_iters=0,
    repair_push_k1_step=0.0,
    use_polygon_verification=False,
    use_worst_case_disturbance=False,
    corridor_radius=0.0,
    obstacle_inflate_radius=0.0,
    verify_disturbance=0.0,
    execution_disturbance=0.0,
    execution_disturbance_seed=None,
    execution_disturbance_mode='step',
    store_verify_results=False,
):
    if planner not in ('standard', 'noerror', 'rtd_rax'):
        raise ValueError("planner must be 'standard', 'noerror', or 'rtd_rax'")

    frs_key = 'standard' if planner == 'standard' else 'noerror'
    frs, fp = models[frs_key]

    agent = TurtlebotAgent()
    agent.reset([float(scenario.start_pose[0]), float(scenario.start_pose[1]), float(scenario.start_pose[2]), v0])
    spacing = compute_turtlebot_point_spacing(agent.footprint, obstacle_buffer)

    exec_dist_bound = _coerce_disturbance_bound(execution_disturbance, name='execution_disturbance')
    exec_dist_mode = str(execution_disturbance_mode).strip().lower()
    if exec_dist_mode not in ('step', 'episode'):
        raise ValueError("execution_disturbance_mode must be 'step' or 'episode'")
    exec_dist_rng = np.random.default_rng(execution_disturbance_seed)
    if exec_dist_mode == 'episode' and np.any(exec_dist_bound > 0.0):
        episode_exec_dist = exec_dist_rng.uniform(-exec_dist_bound, exec_dist_bound, size=4)
    else:
        episode_exec_dist = np.zeros(4, dtype=float)

    step_records = []
    k_prev = np.zeros(2, dtype=float)
    status = 'terminated'
    collision = None

    for step in range(int(max_steps)):
        dist_goal = float(np.linalg.norm(agent.pose[:2] - scenario.goal.ravel()))
        if dist_goal <= goal_tol and agent.speed <= speed_tol:
            status = 'goal_reached'
            break

        compute_start = time.perf_counter()
        feasible, k_plan, res, solve_timing = _solve_step(
            frs,
            fp,
            agent.state[:, -1],
            agent.speed,
            scenario.goal,
            spacing,
            scenario.obstacle_polys,
            k_prev,
            obstacle_buffer,
        )

        record = {
            'step': int(step),
            'feasible': bool(feasible),
            'solve_setup_time': float(solve_timing['solve_setup_time']),
            'solve_optimize_time': float(solve_timing['solve_optimize_time']),
            'solve_time': float(solve_timing['solve_time']),
            'prepare_time': 0.0,
            'initial_verify_time': 0.0,
            'verify_time': 0.0,
            'repair_time': 0.0,
            'repair_solve_time': 0.0,
            'repair_prepare_time': 0.0,
            'repair_verify_time': 0.0,
            'compute_time': 0.0,
            'repair_iters': 0,
            'repair_applied': False,
            'verify_safe': None,
            'initial_verify_safe': None,
            'disturbance_bound': np.zeros(4, dtype=float),
        }

        if not feasible:
            record['compute_time'] = float(time.perf_counter() - compute_start)
            brake_out = _execute_emergency_brake(agent, scenario, agent.speed)
            if brake_out['collision']:
                status = 'collision'
                collision = brake_out['collision_info']
                step_records.append(record)
                break
            step_records.append(record)
            k_prev = np.zeros(2, dtype=float)
            continue

        candidate, prepare_time = _timed_prepare_candidate(frs, agent, k_plan, scenario.patches, corridor_radius=corridor_radius)
        record['prepare_time'] = float(prepare_time)
        record['k'] = candidate['k']
        record['disturbance_bound'] = candidate['disturbance_bound'].copy()
        record['planned_world'] = candidate['z_world'][:2, :].copy()

        if planner == 'rtd_rax':
            verify_polys = scenario.obstacle_polys if use_polygon_verification else None
            vres, verify_time = _timed_verify_candidate(
                candidate,
                agent,
                scenario.patches,
                verify_horizon,
                scenario,
                verify_uncertainty,
                verify_dt,
                obstacle_polys=verify_polys,
                use_worst_case_disturbance=use_worst_case_disturbance,
                corridor_radius=corridor_radius,
                obstacle_inflate_radius=obstacle_inflate_radius,
                verify_disturbance=verify_disturbance,
            )
            record['initial_verify_time'] = float(verify_time)
            record['initial_verify_safe'] = bool(vres['safe'])
            record['verify_safe'] = bool(vres['safe'])
            chosen = candidate
            chosen_verify = vres
            if store_verify_results:
                record['verify_attempts'] = [
                    {
                        'kind': 'initial',
                        'safe': bool(vres['safe']),
                        'planned_world': candidate['z_world'][:2, :].copy(),
                        'verify_result': vres,
                    }
                ]

            if not vres['safe']:
                record['repair_applied'] = True
                k_cur = np.asarray(k_plan, dtype=float)
                repair_start = time.perf_counter()
                for ridx in range(int(repair_max_iters)):
                    record['repair_iters'] = ridx + 1
                    k_try = k_cur.copy()
                    k_try[1] = np.clip(k_try[1] - repair_speed_backoff, -1.0, 1.0)
                    candidate_try, prepare_time = _timed_prepare_candidate(frs, agent, k_try, scenario.patches, corridor_radius=corridor_radius)
                    record['repair_prepare_time'] += float(prepare_time)
                    v_try, verify_time = _timed_verify_candidate(
                        candidate_try,
                        agent,
                        scenario.patches,
                        verify_horizon,
                        scenario,
                        verify_uncertainty,
                        verify_dt,
                        obstacle_polys=verify_polys,
                        use_worst_case_disturbance=use_worst_case_disturbance,
                        corridor_radius=corridor_radius,
                        obstacle_inflate_radius=obstacle_inflate_radius,
                        verify_disturbance=verify_disturbance,
                    )
                    record['repair_verify_time'] += float(verify_time)
                    if store_verify_results:
                        record['verify_attempts'].append(
                            {
                                'kind': 'speed_backoff',
                                'safe': bool(v_try['safe']),
                                'planned_world': candidate_try['z_world'][:2, :].copy(),
                                'verify_result': v_try,
                            }
                        )
                    if v_try['safe']:
                        chosen = candidate_try
                        chosen_verify = v_try
                        break

                    if int(repair_push_iters) > 0 and float(repair_push_k1_step) > 0.0:
                        dpy = float(candidate_try['disturbance_bound'][1])
                        preferred_dir = -1.0 if dpy >= 0.0 else 1.0
                        push_dirs = [preferred_dir, -preferred_dir]
                        for pidx in range(int(repair_push_iters)):
                            delta = float((pidx + 1) * repair_push_k1_step)
                            for push_dir in push_dirs:
                                k_push = k_try.copy()
                                k_push[0] = np.clip(k_push[0] + push_dir * delta, -1.0, 1.0)
                                candidate_push, prepare_time = _timed_prepare_candidate(frs, agent, k_push, scenario.patches, corridor_radius=corridor_radius)
                                record['repair_prepare_time'] += float(prepare_time)
                                v_push, verify_time = _timed_verify_candidate(
                                    candidate_push,
                                    agent,
                                    scenario.patches,
                                    verify_horizon,
                                    scenario,
                                    verify_uncertainty,
                                    verify_dt,
                                    obstacle_polys=verify_polys,
                                    use_worst_case_disturbance=use_worst_case_disturbance,
                                    corridor_radius=corridor_radius,
                                    obstacle_inflate_radius=obstacle_inflate_radius,
                                    verify_disturbance=verify_disturbance,
                                )
                                record['repair_verify_time'] += float(verify_time)
                                if store_verify_results:
                                    record['verify_attempts'].append(
                                        {
                                            'kind': 'lateral_push',
                                            'safe': bool(v_push['safe']),
                                            'planned_world': candidate_push['z_world'][:2, :].copy(),
                                            'verify_result': v_push,
                                        }
                                    )
                                if v_push['safe']:
                                    chosen = candidate_push
                                    chosen_verify = v_push
                                    break
                            if chosen_verify['safe']:
                                break
                        if chosen_verify['safe']:
                            break

                    buf = obstacle_buffer + (ridx + 1) * repair_buffer_step
                    feas_fix, k_fix, _, solve_fix_timing = _solve_step(
                        frs,
                        fp,
                        agent.state[:, -1],
                        agent.speed,
                        scenario.goal,
                        spacing,
                        scenario.obstacle_polys,
                        k_try,
                        buf,
                    )
                    record['repair_solve_time'] += float(solve_fix_timing['solve_time'])
                    if not feas_fix:
                        k_cur = k_try
                        continue

                    candidate_fix, prepare_time = _timed_prepare_candidate(frs, agent, k_fix, scenario.patches, corridor_radius=corridor_radius)
                    record['repair_prepare_time'] += float(prepare_time)
                    v_fix, verify_time = _timed_verify_candidate(
                        candidate_fix,
                        agent,
                        scenario.patches,
                        verify_horizon,
                        scenario,
                        verify_uncertainty,
                        verify_dt,
                        obstacle_polys=verify_polys,
                        use_worst_case_disturbance=use_worst_case_disturbance,
                        corridor_radius=corridor_radius,
                        obstacle_inflate_radius=obstacle_inflate_radius,
                        verify_disturbance=verify_disturbance,
                    )
                    record['repair_verify_time'] += float(verify_time)
                    if store_verify_results:
                        record['verify_attempts'].append(
                            {
                                'kind': 'buffer_replan',
                                'safe': bool(v_fix['safe']),
                                'planned_world': candidate_fix['z_world'][:2, :].copy(),
                                'verify_result': v_fix,
                            }
                        )
                    k_cur = np.asarray(k_fix, dtype=float)
                    if v_fix['safe']:
                        chosen = candidate_fix
                        chosen_verify = v_fix
                        break

                record['repair_time'] = float(time.perf_counter() - repair_start)

                if not chosen_verify['safe']:
                    record['verify_time'] = float(record['initial_verify_time'] + record['repair_verify_time'])
                    record['verify_safe'] = False
                    record['compute_time'] = float(time.perf_counter() - compute_start)
                    step_records.append(record)
                    brake_out = _execute_emergency_brake(agent, scenario, agent.speed)
                    if brake_out['collision']:
                        status = 'collision'
                        collision = brake_out['collision_info']
                        break
                    k_prev = np.zeros(2, dtype=float)
                    continue

            record['verify_time'] = float(record['initial_verify_time'] + record['repair_verify_time'])
            record['verify_safe'] = bool(chosen_verify['safe'])
            record['disturbance_bound'] = chosen['disturbance_bound'].copy()
            record['planned_world'] = chosen['z_world'][:2, :].copy()
            record['k'] = chosen['k']
            if store_verify_results:
                record['verify_result'] = chosen_verify
            exec_candidate = chosen
        else:
            exec_candidate = candidate

        record['compute_time'] = float(time.perf_counter() - compute_start)
        t_exec_ref, u_exec, z_exec = truncate_reference(
            exec_candidate['t_ref'],
            exec_candidate['u_ref'],
            exec_candidate['z_ref'],
            min(float(t_move), float(exec_candidate['t_ref'][-1])),
        )

        seg_start = agent.state.shape[1] - 1
        # Sample either per-step or per-episode random execution disturbance.
        if exec_dist_mode == 'episode':
            step_exec_dist = episode_exec_dist
        elif np.any(exec_dist_bound > 0.0):
            step_exec_dist = exec_dist_rng.uniform(-exec_dist_bound, exec_dist_bound, size=4)
        else:
            step_exec_dist = np.zeros(4, dtype=float)
        agent.move(
            float(t_exec_ref[-1]),
            t_exec_ref,
            u_exec,
            z_exec,
            disturbance=lambda t, z, _d=step_exec_dist: disturbance_from_patches(z, scenario.patches) + _d,
        )
        segment = agent.state[:, seg_start:]
        hit_idx, collision = _first_collision(segment, scenario, agent.footprint)
        if hit_idx is not None:
            gidx = seg_start + int(hit_idx)
            agent.state = agent.state[:, :gidx + 1]
            agent.time = agent.time[:gidx + 1]
            status = 'collision'
            record['segment_start'] = int(seg_start)
            step_records.append(record)
            break

        goal_idx = _first_goal_hit(segment, scenario.goal, goal_tol)
        if goal_idx is not None:
            gidx = seg_start + int(goal_idx)
            agent.state = agent.state[:, :gidx + 1]
            agent.time = agent.time[:gidx + 1]
            status = 'goal_reached'
            record['segment_start'] = int(seg_start)
            step_records.append(record)
            break

        record['segment_start'] = int(seg_start)
        step_records.append(record)
        k_prev = np.asarray(record['k'], dtype=float)

    return {
        'planner': planner,
        'label': {'standard': 'Standard RTD', 'noerror': 'Non-Inflated RTD', 'rtd_rax': 'RTD-RAX'}[planner],
        'status': status,
        'agent': agent,
        'scenario': scenario,
        'obstacle_buffer': float(obstacle_buffer),
        'collision': collision,
        'step_records': step_records,
        'solve_setup_times': np.array([r['solve_setup_time'] for r in step_records], dtype=float),
        'solve_optimize_times': np.array([r['solve_optimize_time'] for r in step_records], dtype=float),
        'solve_times': np.array([r['solve_time'] for r in step_records], dtype=float),
        'prepare_times': np.array([r['prepare_time'] for r in step_records], dtype=float),
        'initial_verify_times': np.array([r['initial_verify_time'] for r in step_records], dtype=float),
        'verify_times': np.array([r['verify_time'] for r in step_records], dtype=float),
        'repair_times': np.array([r['repair_time'] for r in step_records], dtype=float),
        'compute_times': np.array([r['compute_time'] for r in step_records], dtype=float),
        'repair_count': int(sum(1 for r in step_records if r['repair_applied'])),
        'goal_distance_final': float(np.linalg.norm(agent.pose[:2] - scenario.goal.ravel())),
        'path_arclength': float(compute_path_arclength({'agent': agent})),
    }


def summarize_status_counts(results):
    counts = {'goal_reached': 0, 'collision': 0, 'fail_safe_stop': 0, 'terminated': 0}
    for res in results:
        key = res['status'] if res['status'] in counts else 'terminated'
        counts[key] += 1
    return counts


def summarize_compute_times(results):
    vals = np.concatenate([res['compute_times'] for res in results if res['compute_times'].size > 0], axis=0)
    if vals.size == 0:
        return {
            'count': 0,
            'mean_ms': np.nan,
            'median_ms': np.nan,
            'p95_ms': np.nan,
            'max_ms': np.nan,
        }
    vals_ms = 1e3 * vals
    return {
        'count': int(vals_ms.size),
        'mean_ms': float(np.mean(vals_ms)),
        'median_ms': float(np.median(vals_ms)),
        'p95_ms': float(np.percentile(vals_ms, 95.0)),
        'max_ms': float(np.max(vals_ms)),
    }


def print_result_summary(result):
    comp = summarize_compute_times([result])
    repair_str = f", repairs={result['repair_count']}" if result['repair_count'] > 0 else ''
    print(
        f"[{result['label']}] status={_display_status_text(result['status'])}, steps={len(result['step_records'])}, "
        f"path_arclength={result['path_arclength']:.3f} m{repair_str}, "
        f"final_goal_dist={result['goal_distance_final']:.3f} m, "
        f"mean_compute={comp['mean_ms']:.1f} ms, p95={comp['p95_ms']:.1f} ms"
    )


def _draw_static_world(ax, scenario):
    for idx, poly in enumerate(scenario.obstacle_polys):
        poly_n = _normalize_polygon(poly)
        patch = mpatches.Polygon(
            poly_n[:, :-1].T,
            closed=True,
            facecolor='#b63c3c',
            edgecolor='black',
            linewidth=1.0,
            alpha=0.88,
            label='Obstacle' if idx == 0 else '_nolegend_',
        )
        ax.add_patch(patch)

    for idx, patch in enumerate(scenario.patches):
        rect = patch.rect
        label = 'Disturbance / Ice Patch' if idx == 0 else '_nolegend_'
        rect_patch = mpatches.Rectangle(
            (rect[0], rect[2]),
            rect[1] - rect[0],
            rect[3] - rect[2],
            facecolor='#5fa6d8',
            edgecolor='#255a7a',
            linewidth=1.0,
            alpha=0.18,
            label=label,
        )
        ax.add_patch(rect_patch)

    if scenario.road_half_width is not None:
        ax.axhline(scenario.road_half_width, color='0.35', linestyle='--', linewidth=1.2)
        ax.axhline(-scenario.road_half_width, color='0.35', linestyle='--', linewidth=1.2)
    if scenario.world_bounds is not None:
        x_lo, x_hi, y_lo, y_hi = scenario.world_bounds
        ax.add_patch(
            mpatches.Rectangle(
                (x_lo, y_lo),
                x_hi - x_lo,
                y_hi - y_lo,
                facecolor='none',
                edgecolor='0.35',
                linewidth=1.2,
                linestyle='--',
            )
        )
    ax.plot(float(scenario.start_pose[0]), float(scenario.start_pose[1]), 'ko', markersize=5, label='Start')
    ax.plot(float(scenario.goal[0, 0]), float(scenario.goal[1, 0]), 'k*', markersize=15, label='Goal')
    xlim, ylim = _scenario_plot_limits(scenario)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')


def _draw_collision_emphasis(ax, result, color):
    if result['collision'] is None:
        return

    x_c, y_c = result['collision']['point']
    ax.plot(x_c, y_c, marker='x', color='black', markersize=10, mew=2.2)
    ax.add_patch(
        mpatches.Circle(
            (float(x_c), float(y_c)),
            result['agent'].footprint,
            facecolor='none',
            edgecolor='black',
            linewidth=2.2,
            zorder=6,
        )
    )


def _make_compare_legend(fig, result_a, result_b, anchor):
    results = [result_a, result_b]
    handles = [
        mpatches.Patch(facecolor='#b63c3c', edgecolor='black', alpha=0.88, label='Obstacle'),
    ]
    has_patches = any(len(r['scenario'].patches) > 0 for r in results)
    if has_patches:
        handles.append(
            mpatches.Patch(facecolor='#5fa6d8', edgecolor='#255a7a', alpha=0.18, label='Disturbance patch'),
        )
    handles.extend([
        Line2D([0], [0], marker='o', color='black', linestyle='None', markersize=5, label='Start'),
        Line2D([0], [0], marker='*', color='black', linestyle='None', markersize=13, label='Goal'),
    ])
    for r in results:
        color = _COMPARE_COLORS.get(r['label'], '#333333')
        handles.append(Line2D([0], [0], color=color, linewidth=2.6, label=f"{r['label']} path"))

    if any(r['collision'] is not None for r in results):
        handles.append(
            Line2D(
                [0], [0], marker='o', markerfacecolor='none', markeredgecolor='black',
                markeredgewidth=2.0, color='black', linestyle='None', markersize=12,
                label='Collision footprint',
            ),
        )

    fig.legend(
        handles=handles,
        loc='upper left',
        bbox_to_anchor=anchor,
        frameon=True,
        borderaxespad=0.0,
    )


def plot_compare_episodes(scenario, standard_result, rax_result, save_path=None, show_legend=True):
    apply_case_study_style()
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 5.8), sharex=True, sharey=True)
    results = [standard_result, rax_result]

    for ax, res in zip(axes, results):
        color = _COMPARE_COLORS.get(res['label'], '#333333')
        _draw_static_world(ax, scenario)
        st = res['agent'].state
        ax.plot(st[0, :], st[1, :], color=color, linewidth=2.6, label=res['label'])
        ax.add_patch(
            mpatches.Circle(
                (float(st[0, -1]), float(st[1, -1])),
                res['agent'].footprint,
                facecolor=color,
                edgecolor='black',
                alpha=0.30,
            )
        )
        _draw_collision_emphasis(ax, res, color)
        title_line2 = f"status: {res['status']} | steps: {len(res['step_records'])}"
        if res['repair_count'] > 0:
            title_line2 += f" | repairs: {res['repair_count']}"
        ax.set_title(f"{res['label']}\n{title_line2}")

    if show_legend:
        fig.subplots_adjust(left=0.06, right=0.79, bottom=0.12, top=0.92, wspace=0.18)
        right_ax_pos = axes[1].get_position()
        _make_compare_legend(fig, standard_result, rax_result, anchor=(right_ax_pos.x1 + 0.02, right_ax_pos.y1))
    else:
        fig.subplots_adjust(left=0.06, right=0.96, bottom=0.12, top=0.92, wspace=0.18)
    if save_path is not None:
        ensure_parent_dir(save_path)
        fig.savefig(save_path, bbox_inches='tight')
    return fig


def animate_compare_episodes(scenario, standard_result, rax_result, save_path=None, fps=12, max_frames=240, show_legend=True):
    apply_case_study_style()
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 5.8), sharex=True, sharey=True)
    results = [standard_result, rax_result]

    artists = []
    frame_count_raw = max(r['agent'].state.shape[1] for r in results)
    if frame_count_raw <= max_frames:
        frame_indices = np.arange(frame_count_raw, dtype=int)
    else:
        frame_indices = np.unique(np.linspace(0, frame_count_raw - 1, int(max_frames), dtype=int))

    for ax, res in zip(axes, results):
        color = _COMPARE_COLORS.get(res['label'], '#333333')
        _draw_static_world(ax, scenario)
        ax.set_title(res['label'])
        path_line, = ax.plot([], [], color=color, linewidth=2.6)
        body = mpatches.Circle((0.0, 0.0), res['agent'].footprint, facecolor=color, edgecolor='black', alpha=0.32)
        ax.add_patch(body)
        collision_marker, = ax.plot([], [], marker='x', color='black', markersize=10, mew=2.2, linestyle='None')
        collision_ring = mpatches.Circle((0.0, 0.0), res['agent'].footprint, facecolor='none', edgecolor='black', linewidth=2.2, visible=False)
        ax.add_patch(collision_ring)
        text = ax.text(
            0.02, 0.98, '', transform=ax.transAxes, va='top', ha='left', fontsize=11,
            bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': '0.8'},
        )
        artists.append((res, path_line, body, collision_marker, collision_ring, text))

    if show_legend:
        fig.subplots_adjust(left=0.06, right=0.79, bottom=0.12, top=0.92, wspace=0.18)
        right_ax_pos = axes[1].get_position()
        _make_compare_legend(fig, standard_result, rax_result, anchor=(right_ax_pos.x1 + 0.02, right_ax_pos.y1))
    else:
        fig.subplots_adjust(left=0.06, right=0.96, bottom=0.12, top=0.92, wspace=0.18)

    def update(frame_idx):
        out = []
        for res, path_line, body, collision_marker, collision_ring, text in artists:
            st = res['agent'].state
            idx = min(frame_idx, st.shape[1] - 1)
            path_line.set_data(st[0, :idx + 1], st[1, :idx + 1])
            body.center = (float(st[0, idx]), float(st[1, idx]))
            status_text = res['status'] if idx == st.shape[1] - 1 else 'executing'
            if res['collision'] is not None and idx == st.shape[1] - 1:
                collision_marker.set_data([res['collision']['point'][0]], [res['collision']['point'][1]])
                collision_ring.center = (float(res['collision']['point'][0]), float(res['collision']['point'][1]))
                collision_ring.set_visible(True)
            else:
                collision_marker.set_data([], [])
                collision_ring.set_visible(False)
            info = f"t = {res['agent'].time[idx]:.2f} s\nstatus: {status_text}\nsteps: {len(res['step_records'])}"
            if res['repair_count'] > 0:
                info += f" | repairs: {res['repair_count']}"
            text.set_text(info)
            out.extend([path_line, body, collision_marker, collision_ring, text])
        return out

    anim = FuncAnimation(
        fig, update, frames=frame_indices, interval=1000.0 / max(float(fps), 1.0), blit=False,
    )
    if save_path is not None:
        ensure_parent_dir(save_path)
        anim.save(save_path, writer=PillowWriter(fps=fps))
    return fig, anim


_COMPARE_COLORS = {
    'Standard RTD': '#8e1f1f',
    'RTD-RAX': '#1f6f3d',
}

_TRIPLE_COLORS = {
    'Standard RTD': '#8e1f1f',
    'Non-Inflated RTD': '#c47a21',
    'RTD-RAX': '#1f6f3d',
}

_TRIPLE_FRS_COLOR = [0.3, 0.8, 0.5]
_TRIPLE_BUF_COLOR = [1.0, 0.55, 0.55]
_TRIPLE_PTS_COLOR = [0.4, 0.05, 0.05]


def _build_frame_to_step(result):
    n_frames = result['agent'].state.shape[1]
    boundaries = []
    for idx, rec in enumerate(result['step_records']):
        if 'segment_start' in rec:
            boundaries.append((int(rec['segment_start']), idx))
    mapping = [-1] * n_frames
    for frame_idx in range(n_frames):
        for seg_start, step_idx in boundaries:
            if seg_start <= frame_idx:
                mapping[frame_idx] = step_idx
    return mapping


def _planner_frs_key(result):
    return 'standard' if result['planner'] == 'standard' else 'noerror'


def _first_available_index(values):
    for idx, value in enumerate(values):
        if value is not None:
            return idx
    return None


def _last_available_index(values):
    for idx in range(len(values) - 1, -1, -1):
        if values[idx] is not None:
            return idx
    return None


def _display_status_text(status_text):
    text = str(status_text).strip().replace('_', ' ')
    if text == 'fail safe stop':
        return 'fail-safe stop'
    return text


def _status_color(status_text, is_final=True):
    if not is_final:
        return 'black'
    text = _display_status_text(status_text).lower()
    if text == 'goal reached':
        return 'green'
    if text in ('terminated', 'collision', 'fail-safe stop', 'fail-safe'):
        return 'red'
    return 'black'


def _metric_line(result, status_text, repair_count=None):
    parts = [f"status: {_display_status_text(status_text)}", f"steps: {len(result['step_records'])}"]
    if repair_count is None:
        repair_count = (result['repair_count'] if result['repair_count'] > 0 else None)
    if repair_count is not None:
        parts.append(f"repairs: {int(repair_count)}")
    return ' | '.join(parts)


def _step_metric_line(result, step_count=None, repair_count=None):
    count = len(result['step_records']) if step_count is None else int(step_count)
    parts = [f"steps: {count}"]
    if repair_count is None:
        repair_count = (result['repair_count'] if result['repair_count'] > 0 else None)
    if repair_count is not None:
        parts.append(f"repairs: {int(repair_count)}")
    return ' | '.join(parts)


def _completed_step_count_for_frame(result, frame_idx, include_current_boundary=False):
    idx = int(frame_idx)
    completed = 0
    for rec in result['step_records']:
        if 'segment_start' not in rec:
            continue
        seg_start = int(rec['segment_start'])
        if seg_start < idx or (include_current_boundary and seg_start <= idx):
            completed += 1
    return completed


def _completed_repair_count(result, step_count):
    return int(sum(1 for rec in result['step_records'][: max(int(step_count), 0)] if rec.get('repair_applied', False)))


def _make_verify_tube_rect(ax, row, frac, cmap_name='Blues', visible=False):
    x_lo, x_hi, y_lo, y_hi = np.asarray(row, dtype=float)
    cmap = plt.get_cmap(cmap_name)
    col = cmap(0.30 + 0.60 * float(frac))
    rect = mpatches.FancyBboxPatch(
        (x_lo, y_lo),
        x_hi - x_lo,
        y_hi - y_lo,
        boxstyle='square,pad=0',
        linewidth=0,
        facecolor=col,
        alpha=0.12,
        visible=bool(visible),
        zorder=3,
    )
    ax.add_patch(rect)
    return rect


def _build_verify_tube_animation_data(ax, result):
    all_rects = []
    step_ranges = []
    rect_idx = 0
    for rec in result['step_records']:
        vr = rec.get('verify_result')
        if vr is None:
            step_ranges.append(None)
            continue
        xy_tube = np.asarray(vr.get('xy_tube', np.zeros((0, 4))), dtype=float)
        start = rect_idx
        n_tube = max(len(xy_tube), 1)
        for idx, row in enumerate(xy_tube):
            rect = _make_verify_tube_rect(
                ax,
                row,
                frac=idx / max(n_tube - 1, 1),
                visible=False,
            )
            all_rects.append(rect)
            rect_idx += 1
        step_ranges.append((start, rect_idx))
    return all_rects, step_ranges


def _set_verify_tube_visibility(rects, step_ranges, step_idx):
    for rect in rects:
        rect.set_visible(False)
    if 0 <= step_idx < len(step_ranges):
        bounds = step_ranges[step_idx]
        if bounds is not None:
            start, end = bounds
            for rect in rects[start:end]:
                rect.set_visible(True)


def _draw_verify_tube_snapshot(ax, result):
    step_idx = _last_available_index(
        [rec.get('verify_result') for rec in result['step_records']]
    )
    if step_idx is None:
        return False
    vr = result['step_records'][step_idx]['verify_result']
    xy_tube = np.asarray(vr.get('xy_tube', np.zeros((0, 4))), dtype=float)
    n_tube = max(len(xy_tube), 1)
    for idx, row in enumerate(xy_tube):
        _make_verify_tube_rect(
            ax,
            row,
            frac=idx / max(n_tube - 1, 1),
            visible=True,
        )
    return xy_tube.shape[0] > 0


def _draw_obs_display(ax, obs_bufs, obs_pts):
    for obs_buf in obs_bufs:
        if obs_buf is not None and obs_buf.shape[1] > 0:
            ax.fill(obs_buf[0], obs_buf[1], color=_TRIPLE_BUF_COLOR, alpha=0.5, zorder=1)
    for obs_pts_i in obs_pts:
        if obs_pts_i is not None and obs_pts_i.shape[1] > 0:
            ax.plot(obs_pts_i[0], obs_pts_i[1], '.', color=_TRIPLE_PTS_COLOR, markersize=5, zorder=3)


def _tighten_triple_axes(fig, axes, show_legend):
    axes = list(axes)
    for idx, ax in enumerate(axes):
        ax.set_xlabel('')
        if idx == 0:
            ax.set_ylabel('y [m]')
        else:
            ax.set_ylabel('')
    fig.supxlabel('x [m]')
    if show_legend:
        fig.subplots_adjust(left=0.055, right=0.80, bottom=0.12, top=0.92, wspace=0.04)
    else:
        fig.subplots_adjust(left=0.055, right=0.985, bottom=0.12, top=0.92, wspace=0.04)


def _add_status_box(ax, result, status_text, step_count=None):
    count = len(result['step_records']) if step_count is None else int(step_count)
    text = ax.text(
        0.02,
        0.98,
        f"status: {_display_status_text(status_text)}\n{_step_metric_line(result, count)}",
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=11,
        color=_status_color(status_text, is_final=True),
        bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': '0.8'},
        zorder=10,
    )
    return text


def _make_triple_compare_legend(fig, results, anchor, show_obs_display=False, show_frs=False, show_mmr=False):
    handles = [
        mpatches.Patch(facecolor='#b63c3c', edgecolor='black', alpha=0.88, label='Obstacle'),
    ]
    if show_obs_display:
        handles.append(mpatches.Patch(facecolor=_TRIPLE_BUF_COLOR, alpha=0.5, label='Buffered Obstacle'))
        handles.append(
            Line2D([0], [0], marker='.', color=_TRIPLE_PTS_COLOR, linestyle='None', markersize=5, label='Obstacle Discretization')
        )
    if any(len(r['scenario'].patches) > 0 for r in results):
        handles.append(
            mpatches.Patch(facecolor='#5fa6d8', edgecolor='#255a7a', alpha=0.18, label='Disturbance patch'),
        )
    handles.extend([
        Line2D([0], [0], marker='o', color='black', linestyle='None', markersize=5, label='Start'),
        Line2D([0], [0], marker='*', color='black', linestyle='None', markersize=13, label='Goal'),
    ])
    for r in results:
        color = _TRIPLE_COLORS.get(r['label'], '#333333')
        handles.append(Line2D([0], [0], color=color, linewidth=2.6, label=f"{r['label']} path"))
    if show_frs:
        handles.append(Line2D([0], [0], color=_TRIPLE_FRS_COLOR, linewidth=1.8, label='FRS @ k_opt'))
    if show_mmr:
        handles.append(Line2D([0], [0], color=plt.get_cmap('Blues')(0.92), alpha=0.9, linewidth=2.0, label='MMR FRS'))
    if any(r['collision'] is not None for r in results):
        handles.append(
            Line2D(
                [0], [0], marker='o', markerfacecolor='none', markeredgecolor='black',
                markeredgewidth=2.0, color='black', linestyle='None', markersize=12,
                label='Collision footprint',
            ),
        )
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=anchor, frameon=True, borderaxespad=0.0)


def plot_triple_compare_episodes(scenario, result_a, result_b, result_c, models=None, save_path=None, show_legend=True):
    apply_case_study_style()
    fig, axes = plt.subplots(1, 3, figsize=(18.8, 5.8), sharex=True, sharey=True)
    results = [result_a, result_b, result_c]
    contour_sets = [None] * len(results)
    obs_displays = [None] * len(results)
    show_frs = False
    show_mmr = False
    show_obs_display = False

    if models is not None:
        contour_sets = [
            compute_step_contours(res, models[_planner_frs_key(res)][0], grid_res=150)
            for res in results
        ]
        show_frs = any(
            res['planner'] != 'rtd_rax' and _last_available_index(contours or []) is not None
            for res, contours in zip(results, contour_sets)
        )
        obs_displays = [
            _precompute_obs_display(
                scenario,
                models[_planner_frs_key(res)][0],
                res['agent'].state[:, 0],
                res['agent'].footprint,
                res['obstacle_buffer'],
            ) if res['planner'] in ('standard', 'noerror') else None
            for res in results
        ]
        show_obs_display = any(display is not None for display in obs_displays)

    for ax, res, contours, obs_display in zip(axes, results, contour_sets, obs_displays):
        color = _TRIPLE_COLORS.get(res['label'], '#333333')
        _draw_static_world(ax, scenario)
        if obs_display is not None:
            _draw_obs_display(ax, *obs_display)
        contour_idx = _last_available_index(contours or [])
        if contour_idx is not None and res['planner'] != 'rtd_rax':
            contour = contours[contour_idx]
            ax.plot(contour[0], contour[1], color=_TRIPLE_FRS_COLOR, linewidth=1.8, zorder=4)
        if res['planner'] == 'rtd_rax':
            show_mmr = _draw_verify_tube_snapshot(ax, res) or show_mmr
        st = res['agent'].state
        ax.plot(st[0, :], st[1, :], color=color, linewidth=2.6, label=res['label'], zorder=5)
        ax.add_patch(
            mpatches.Circle(
                (float(st[0, -1]), float(st[1, -1])),
                res['agent'].footprint,
                facecolor=color,
                edgecolor='black',
                alpha=0.30,
                zorder=6,
            )
        )
        _draw_collision_emphasis(ax, res, color)
        ax.set_title(res['label'])
        _add_status_box(ax, res, res['status'])

    _tighten_triple_axes(fig, axes, show_legend=show_legend)
    if show_legend:
        right_ax_pos = axes[-1].get_position()
        _make_triple_compare_legend(
            fig,
            results,
            anchor=(right_ax_pos.x1 + 0.02, right_ax_pos.y1),
            show_obs_display=show_obs_display,
            show_frs=show_frs,
            show_mmr=show_mmr,
        )
    if save_path is not None:
        ensure_parent_dir(save_path)
        fig.savefig(save_path, bbox_inches='tight')
    return fig


def animate_triple_compare_episodes(
    scenario, result_a, result_b, result_c, models=None, save_path=None, fps=12, max_frames=240, show_legend=True,
):
    apply_case_study_style()
    fig, axes = plt.subplots(1, 3, figsize=(18.8, 5.8), sharex=True, sharey=True)
    results = [result_a, result_b, result_c]
    contour_sets = [None] * len(results)
    obs_displays = [None] * len(results)
    frame_to_steps = [_build_frame_to_step(res) for res in results]
    show_frs = False
    show_mmr = False
    show_obs_display = False

    if models is not None:
        contour_sets = [
            compute_step_contours(res, models[_planner_frs_key(res)][0], grid_res=150)
            for res in results
        ]
        show_frs = any(
            res['planner'] != 'rtd_rax' and _last_available_index(contours or []) is not None
            for res, contours in zip(results, contour_sets)
        )
        obs_displays = [
            _precompute_obs_display(
                scenario,
                models[_planner_frs_key(res)][0],
                res['agent'].state[:, 0],
                res['agent'].footprint,
                res['obstacle_buffer'],
            ) if res['planner'] in ('standard', 'noerror') else None
            for res in results
        ]
        show_obs_display = any(display is not None for display in obs_displays)

    artists = []
    frame_count_raw = max(r['agent'].state.shape[1] for r in results)
    if frame_count_raw <= max_frames:
        frame_indices = np.arange(frame_count_raw, dtype=int)
    else:
        frame_indices = np.unique(np.linspace(0, frame_count_raw - 1, int(max_frames), dtype=int))

    for ax, res, contours, f2s, obs_display in zip(axes, results, contour_sets, frame_to_steps, obs_displays):
        color = _TRIPLE_COLORS.get(res['label'], '#333333')
        _draw_static_world(ax, scenario)
        if obs_display is not None:
            _draw_obs_display(ax, *obs_display)
        ax.set_title(res['label'])
        path_line, = ax.plot([], [], color=color, linewidth=2.6, zorder=5)
        body = mpatches.Circle((0.0, 0.0), res['agent'].footprint, facecolor=color, edgecolor='black', alpha=0.32, zorder=6)
        ax.add_patch(body)
        frs_line, = ax.plot([], [], color=_TRIPLE_FRS_COLOR, linewidth=1.8, zorder=4)
        collision_marker, = ax.plot([], [], marker='x', color='black', markersize=10, mew=2.2, linestyle='None')
        collision_ring = mpatches.Circle(
            (0.0, 0.0), res['agent'].footprint, facecolor='none', edgecolor='black', linewidth=2.2, visible=False,
        )
        ax.add_patch(collision_ring)
        mmr_rects = []
        mmr_step_ranges = []
        if res['planner'] == 'rtd_rax':
            mmr_rects, mmr_step_ranges = _build_verify_tube_animation_data(ax, res)
            show_mmr = bool(mmr_rects) or show_mmr
        text = ax.text(
            0.02, 0.98, '', transform=ax.transAxes, va='top', ha='left', fontsize=11,
            bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': '0.8'},
        )
        artists.append((res, path_line, body, frs_line, collision_marker, collision_ring, text, contours, f2s, mmr_rects, mmr_step_ranges))

    _tighten_triple_axes(fig, axes, show_legend=show_legend)
    if show_legend:
        right_ax_pos = axes[-1].get_position()
        _make_triple_compare_legend(
            fig,
            results,
            anchor=(right_ax_pos.x1 + 0.02, right_ax_pos.y1),
            show_obs_display=show_obs_display,
            show_frs=show_frs,
            show_mmr=show_mmr,
        )

    def update(frame_idx):
        out = []
        for res, path_line, body, frs_line, collision_marker, collision_ring, text, contours, f2s, mmr_rects, mmr_step_ranges in artists:
            st = res['agent'].state
            idx = min(frame_idx, st.shape[1] - 1)
            path_line.set_data(st[0, :idx + 1], st[1, :idx + 1])
            body.center = (float(st[0, idx]), float(st[1, idx]))
            step_idx = f2s[idx] if idx < len(f2s) else -1
            if (
                res['planner'] != 'rtd_rax'
                and 0 <= step_idx < len(contours or [])
                and contours[step_idx] is not None
            ):
                contour = contours[step_idx]
                frs_line.set_data(contour[0], contour[1])
            else:
                frs_line.set_data([], [])
            if mmr_rects:
                _set_verify_tube_visibility(mmr_rects, mmr_step_ranges, step_idx)
            is_final = idx == st.shape[1] - 1
            status_text = res['status'] if is_final else 'executing'
            if res['collision'] is not None and is_final:
                collision_marker.set_data([res['collision']['point'][0]], [res['collision']['point'][1]])
                collision_ring.center = (float(res['collision']['point'][0]), float(res['collision']['point'][1]))
                collision_ring.set_visible(True)
            else:
                collision_marker.set_data([], [])
                collision_ring.set_visible(False)
            current_step_count = _completed_step_count_for_frame(res, idx, include_current_boundary=False)
            current_repair_count = (
                _completed_repair_count(res, current_step_count)
                if res['planner'] == 'rtd_rax' and res['repair_count'] > 0
                else None
            )
            text.set_color(_status_color(status_text, is_final=is_final))
            text.set_text(
                f"t = {res['agent'].time[idx]:.2f} s\n"
                f"status: {_display_status_text(status_text)}\n"
                f"{_step_metric_line(res, current_step_count, repair_count=current_repair_count)}"
            )
            out.extend([path_line, body, frs_line, collision_marker, collision_ring, text, *mmr_rects])
        return out

    anim = FuncAnimation(fig, update, frames=frame_indices, interval=1000.0 / max(float(fps), 1.0), blit=False)
    if save_path is not None:
        ensure_parent_dir(save_path)
        anim.save(save_path, writer=PillowWriter(fps=fps))
    return fig, anim


def _sample_history(items, max_count):
    items = list(items)
    if len(items) <= int(max_count):
        return items
    idx = np.unique(np.linspace(0, len(items) - 1, int(max_count), dtype=int))
    return [items[int(i)] for i in idx]


def _make_repair_view_item(step, kind, safe, verify_result, planned_world):
    trace_xy = np.asarray(verify_result.get('nom_xy', np.zeros((0, 2))), dtype=float)
    if trace_xy.shape[0] == 0:
        planned_world = np.asarray(planned_world, dtype=float)
        if planned_world.shape[1] > 0:
            trace_xy = planned_world.T
    return {
        'step': int(step),
        'kind': str(kind),
        'safe': bool(safe),
        'trace_xy': trace_xy,
        'xy_tube': np.asarray(verify_result.get('xy_tube', np.zeros((0, 4))), dtype=float),
    }


def _collect_repair_view_history(result):
    safe_history = []
    unsafe_history = []
    for rec in result['step_records']:
        attempts = list(rec.get('verify_attempts', []))
        if attempts:
            first_attempt = attempts[0]
            vr_first = first_attempt.get('verify_result')
            if vr_first is not None and not bool(first_attempt.get('safe', False)):
                unsafe_history.append(
                    _make_repair_view_item(
                        rec['step'],
                        'initial_rejected',
                        False,
                        vr_first,
                        first_attempt.get('planned_world', np.zeros((2, 0))),
                    )
                )
        vr_final = rec.get('verify_result')
        if vr_final is not None and bool(rec.get('verify_safe', False)):
            item = _make_repair_view_item(
                rec['step'],
                'final_safe',
                True,
                vr_final,
                rec.get('planned_world', np.zeros((2, 0))),
            )
            item['is_replacement'] = bool(rec.get('repair_applied', False))
            safe_history.append(item)
    return safe_history, unsafe_history


def _repair_view_limits(scenario, result, safe_history, unsafe_history):
    xlim, ylim = _scenario_plot_limits(scenario)
    y_lo, y_hi = [float(v) for v in ylim]
    y_candidates = [y_lo]
    st = result['agent'].state
    if st.shape[1] > 0:
        y_candidates.append(float(np.min(st[1, :]) - result['agent'].footprint - 0.08))
    for item in list(safe_history) + list(unsafe_history):
        trace_xy = np.asarray(item['trace_xy'], dtype=float)
        xy_tube = np.asarray(item['xy_tube'], dtype=float)
        if trace_xy.shape[0] > 0:
            y_candidates.append(float(np.min(trace_xy[:, 1]) - 0.08))
        if xy_tube.shape[0] > 0:
            y_candidates.append(float(np.min(xy_tube[:, 2]) - 0.08))
    return xlim, (min(y_candidates) - 0.06, y_hi)


def _repair_view_style(item):
    if not item['safe']:
        return {
            'edgecolor': '#ea8a8a',
            'facecolor': 'none',
            'linewidth': 1.55,
            'alpha': 0.60,
            'zorder': 8,
            'linestyle': 'solid',
            'hatch': None,
        }
    if bool(item.get('is_replacement', False)):
        return {
            'edgecolor': '#7bcf88',
            'facecolor': 'none',
            'linewidth': 1.9,
            'alpha': 0.95,
            'zorder': 9,
            'linestyle': 'solid',
            'hatch': None,
        }
    return {
        'edgecolor': '#1f5ea8',
        'facecolor': 'none',
        'linewidth': 1.0,
        'alpha': 0.22,
        'zorder': 6,
        'linestyle': 'solid',
        'hatch': None,
    }


def _make_tube_rect(row, style, visible=True):
    x0, x1, y0, y1 = [float(v) for v in np.asarray(row, dtype=float)]
    return mpatches.Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        facecolor=style['facecolor'],
        edgecolor=style['edgecolor'],
        linewidth=style['linewidth'],
        alpha=style['alpha'],
        linestyle=style['linestyle'],
        hatch=style['hatch'],
        visible=bool(visible),
        zorder=style['zorder'],
    )


def _draw_tube_boxes(ax, xy_tube, style, max_boxes=42):
    xy_tube = np.asarray(xy_tube, dtype=float)
    if xy_tube.shape[0] == 0:
        return
    stride = max(1, int(np.ceil(xy_tube.shape[0] / float(max_boxes))))
    for idx in range(0, xy_tube.shape[0], stride):
        ax.add_patch(_make_tube_rect(xy_tube[idx], style, visible=True))


def _build_repair_view_attempt_artists(ax, history):
    artists = []
    for item in history:
        style = _repair_view_style(item)
        if not item['safe']:
            category = 'rejected'
        elif bool(item.get('is_replacement', False)):
            category = 'replacement'
        else:
            category = 'normal'
        rects = []
        xy_tube = np.asarray(item['xy_tube'], dtype=float)
        if xy_tube.shape[0] > 0:
            stride = max(1, int(np.ceil(xy_tube.shape[0] / 42.0)))
            for idx in range(0, xy_tube.shape[0], stride):
                rect = _make_tube_rect(xy_tube[idx], style, visible=False)
                ax.add_patch(rect)
                rects.append(rect)
        artists.append((item, category, rects))
    return artists


def _set_attempt_artist_visibility(artists, step_idx, phase='normal'):
    step_idx = int(step_idx)
    for item, category, rects in artists:
        item_step = int(item['step'])
        visible = False
        if category == 'normal':
            visible = item_step <= step_idx
        elif category == 'replacement':
            if phase == 'show_rejected':
                visible = item_step < step_idx
            else:
                visible = item_step <= step_idx
        elif category == 'rejected':
            visible = (phase == 'show_rejected' and item_step == step_idx)
        for rect in rects:
            rect.set_visible(visible)


def _repair_step_boundaries(result):
    return {
        int(rec['step']): int(rec['segment_start'])
        for rec in result['step_records']
        if 'segment_start' in rec
    }


def _build_repair_view_frames(result, frame_indices, repair_steps, hold_frames):
    repair_steps = {int(s) for s in repair_steps}
    step_boundaries = _repair_step_boundaries(result)
    frames = []
    for idx in frame_indices:
        idx_int = int(idx)
        repair_step_here = None
        for step, seg_start in step_boundaries.items():
            if step in repair_steps and seg_start == idx_int:
                repair_step_here = step
                break
        if repair_step_here is not None:
            for _ in range(int(hold_frames)):
                frames.append((idx_int, 'show_rejected'))
            for _ in range(int(hold_frames)):
                frames.append((idx_int, 'show_replacement'))
        frames.append((idx_int, 'normal'))
    return frames


def _repair_view_legend_handles():
    return [
        mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='#1f5ea8', linewidth=1.5, label='Safe Tube'),
        mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='#ea8a8a', linewidth=1.7, label='Unsafe Tube'),
        mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='#7bcf88', linewidth=1.8, label='Repaired Tube'),
        Line2D([0], [0], color='purple', linestyle='--', linewidth=2.3, label='Executed Path'),
    ]


def _make_repair_view_legend(ax):
    return ax.legend(
        handles=_repair_view_legend_handles(),
        loc='upper right',
        bbox_to_anchor=(0.985, 0.985),
        frameon=True,
        framealpha=0.92,
        borderaxespad=0.0,
    )


def _layout_repair_view(fig, ax, show_legend):
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.92)
    if show_legend:
        _make_repair_view_legend(ax)


def plot_rax_repair_view(scenario, noerror_result, rax_result, models, save_path=None, show_legend=True):
    apply_case_study_style()
    fig, ax = plt.subplots(1, 1, figsize=(10.2, 6.2))
    _draw_static_world(ax, scenario)
    ax.set_title('Angled Obstacle: RTD-RAX')

    safe_history_all, unsafe_history = _collect_repair_view_history(rax_result)
    replacement_history = [item for item in safe_history_all if bool(item.get('is_replacement', False))]
    normal_safe_history = [item for item in safe_history_all if not bool(item.get('is_replacement', False))]
    normal_safe_sample = _sample_history(
        normal_safe_history,
        max_count=max(0, 10 - len(replacement_history)),
    )
    for item in normal_safe_sample:
        _draw_tube_boxes(ax, item['xy_tube'], style=_repair_view_style(item))
    for item in unsafe_history:
        _draw_tube_boxes(ax, item['xy_tube'], style=_repair_view_style(item))
    for item in replacement_history:
        _draw_tube_boxes(ax, item['xy_tube'], style=_repair_view_style(item))

    st = rax_result['agent'].state
    ax.plot(st[0, :], st[1, :], color='purple', linestyle='--', linewidth=2.3, zorder=7)
    ax.plot(st[0, 0], st[1, 0], 'ko', markerfacecolor='none', markersize=8, zorder=10)
    ax.plot(float(scenario.goal[0, 0]), float(scenario.goal[1, 0]), 'k*', markersize=16, zorder=10)
    ax.add_patch(
        mpatches.Circle(
            (float(st[0, -1]), float(st[1, -1])),
            rax_result['agent'].footprint,
            facecolor=_TRIPLE_COLORS['RTD-RAX'],
            edgecolor='black',
            alpha=0.28,
            zorder=10,
        )
    )

    xlim, ylim = _repair_view_limits(scenario, rax_result, safe_history_all, unsafe_history)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

    _layout_repair_view(fig, ax, show_legend)

    if save_path is not None:
        ensure_parent_dir(save_path)
        fig.savefig(save_path, bbox_inches='tight')
    return fig


def animate_rax_repair_view(scenario, noerror_result, rax_result, models, save_path=None, fps=12, max_frames=240, show_legend=False):
    apply_case_study_style()
    fig, ax = plt.subplots(1, 1, figsize=(8.6, 6.2))
    _draw_static_world(ax, scenario)
    ax.set_title('Angled Obstacle: RTD-RAX')

    st = rax_result['agent'].state
    frame_to_step = _build_frame_to_step(rax_result)
    safe_history, unsafe_history = _collect_repair_view_history(rax_result)
    attempt_artists = _build_repair_view_attempt_artists(ax, safe_history + unsafe_history)

    path_line, = ax.plot([], [], color='purple', linestyle='--', linewidth=2.3, zorder=7)
    body = mpatches.Circle((0.0, 0.0), rax_result['agent'].footprint, facecolor=_TRIPLE_COLORS['RTD-RAX'], edgecolor='black', alpha=0.28, zorder=10)
    ax.add_patch(body)
    text = ax.text(
        0.02, 0.98, '', transform=ax.transAxes, va='top', ha='left', fontsize=11,
        bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': '0.8'},
        zorder=11,
    )
    ax.plot(st[0, 0], st[1, 0], 'ko', markerfacecolor='none', markersize=8, zorder=10)
    ax.plot(float(scenario.goal[0, 0]), float(scenario.goal[1, 0]), 'k*', markersize=16, zorder=10)

    xlim, ylim = _repair_view_limits(scenario, rax_result, safe_history, unsafe_history)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

    _layout_repair_view(fig, ax, show_legend)

    frame_count_raw = st.shape[1]
    if frame_count_raw <= max_frames:
        frame_indices = np.arange(frame_count_raw, dtype=int)
    else:
        frame_indices = np.unique(np.linspace(0, frame_count_raw - 1, int(max_frames), dtype=int))
    repair_steps = [item['step'] for item in safe_history if bool(item.get('is_replacement', False))]
    repair_boundaries = list(_repair_step_boundaries(rax_result).values())
    if repair_boundaries:
        frame_indices = np.unique(
            np.concatenate([frame_indices, np.asarray(repair_boundaries, dtype=int)])
        )
    anim_frames = _build_repair_view_frames(
        rax_result,
        frame_indices,
        repair_steps=repair_steps,
        hold_frames=max(3, int(np.ceil(float(fps) * 0.45))),
    )

    def update(frame_info):
        frame_idx, phase = frame_info
        idx = min(int(frame_idx), st.shape[1] - 1)
        step_idx = frame_to_step[idx] if idx < len(frame_to_step) else -1
        path_line.set_data(st[0, :idx + 1], st[1, :idx + 1])
        body.center = (float(st[0, idx]), float(st[1, idx]))
        _set_attempt_artist_visibility(attempt_artists, step_idx, phase=phase)
        current_step_count = _completed_step_count_for_frame(
            rax_result,
            idx,
            include_current_boundary=False,
        )
        repair_step_count = current_step_count
        if phase == 'show_replacement':
            repair_step_count = _completed_step_count_for_frame(
                rax_result,
                idx,
                include_current_boundary=True,
            )
        current_repair_count = _completed_repair_count(rax_result, repair_step_count)
        status_text = rax_result['status'] if idx == st.shape[1] - 1 else 'executing'
        text.set_color(_status_color(status_text, is_final=(idx == st.shape[1] - 1)))
        phase_line = ''
        if phase == 'show_rejected':
            phase_line = '\nrepair event: rejected candidate'
        elif phase == 'show_replacement':
            phase_line = '\nrepair event: replacement selected'
        text.set_text(
            f"t = {rax_result['agent'].time[idx]:.2f} s\n"
            f"status: {_display_status_text(status_text)}\n"
            f"{_step_metric_line(rax_result, current_step_count, repair_count=current_repair_count)}"
            f"{phase_line}"
        )
        out = [path_line, body, text]
        for _item, _category, rects in attempt_artists:
            out.extend(rects)
        return out

    anim = FuncAnimation(fig, update, frames=anim_frames, interval=1000.0 / max(float(fps), 1.0), blit=False)
    if save_path is not None:
        ensure_parent_dir(save_path)
        anim.save(save_path, writer=PillowWriter(fps=fps))
    return fig, anim
