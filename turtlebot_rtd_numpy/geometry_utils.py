"""
geometry_utils.py  –  NumPy version
====================================
Coordinate-frame transforms, polygon buffering / discretization, and helper
geometry for the Turtlebot RTD pipeline.

Direct translations of:
  RTD/utility/geometry/world_to_local.m
  RTD/utility/geometry/FRS_to_world.m
  RTD/utility/geometry/world_to_FRS.m
  RTD/utility/geometry/buffer_polygon_obstacles.m
  RTD/utility/geometry/interpolate_polyline_with_spacing.m
  RTD/utility/geometry/crop_points_outside_region.m
  simulator/src/utility/geometry/shape_creation/make_random_polygon.m
  RTD_tutorial/step_4_online_planning/functions/compute_turtlebot_point_spacings.m
  RTD_tutorial/step_4_online_planning/functions/compute_turtlebot_discretized_obs.m
"""

import numpy as np
try:
    from shapely.geometry import Polygon as ShapelyPolygon
    _SHAPELY = True
except ImportError:
    _SHAPELY = False


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def world_to_local(robot_pose, P_world):
    """Transform points from the world frame to the robot's local frame.

    Args:
        robot_pose: array-like of length >= 3  [x, y, heading, ...]
        P_world:    (2, N) or (3, N) points in world frame

    Returns:
        P_out: same shape as P_world, expressed in the robot's local frame
    """
    x = float(robot_pose[0])
    y = float(robot_pose[1])
    h = float(robot_pose[2])

    P_out = np.array(P_world, dtype=float)

    # Translate to robot position
    P_out[0, :] -= x
    P_out[1, :] -= y

    # Rotate by -h  (R_inv = [[cos h, sin h], [-sin h, cos h]])
    c, s = np.cos(h), np.sin(h)
    R = np.array([[ c, s],
                  [-s, c]])
    P_out[:2, :] = R @ P_out[:2, :]

    # If a heading row is present, shift it
    if P_world.shape[0] > 2:
        P_out[2, :] -= h

    return P_out


def FRS_to_world(P_FRS, pose, x0, y0, D):
    """Transform points from the FRS frame to the world frame.

    The FRS is offset by (x0, y0) and scaled by D.

    Args:
        P_FRS: (2, N) points in the FRS coordinate frame
        pose:  array-like [x, y, heading, ...]
        x0:    FRS x-origin offset
        y0:    FRS y-origin offset
        D:     distance scale (FRS units → metres)

    Returns:
        P_world: (2, N) points in the world frame
    """
    x = float(pose[0])
    y = float(pose[1])
    h = float(pose[2])

    # Remove FRS offset
    P = P_FRS - np.array([[x0], [y0]])

    # Unscale
    P = D * P

    # Rotate to robot heading
    c, s = np.cos(h), np.sin(h)
    R = np.array([[c, -s],
                  [s,  c]])
    P = R @ P

    # Translate to robot world position
    P = P + np.array([[x], [y]])

    return P


def world_to_FRS(P_world, pose, x0, y0, D):
    """Transform points from the world frame to the FRS frame (inverse of FRS_to_world).

    Args:
        P_world: (2, N) points in the world frame
        pose:    array-like [x, y, heading, ...]
        x0:      FRS x-origin offset
        y0:      FRS y-origin offset
        D:       distance scale (metres → FRS units)

    Returns:
        P_FRS: (2, N) points in the FRS coordinate frame
    """
    x = float(pose[0])
    y = float(pose[1])
    h = float(pose[2])

    # Translate to robot position
    P = P_world - np.array([[x], [y]])

    # Rotate by -h  (inverse rotation)
    c, s = np.cos(h), np.sin(h)
    R_inv = np.array([[ c, s],
                      [-s, c]])
    P = R_inv @ P

    # Scale down
    P = P / D

    # Add FRS offset
    P = P + np.array([[x0], [y0]])

    return P


# ---------------------------------------------------------------------------
# Obstacle cropping
# ---------------------------------------------------------------------------

def crop_points_outside_region(cx, cy, P, L):
    """Keep only points inside the [-L, L]^2 box centred at (cx, cy).

    Points with NaN coordinates are also removed.

    Args:
        cx, cy: centre coordinates
        P:      (2, N) array
        L:      half-side-length of the box

    Returns:
        P_keep: (2, M) with M <= N
    """
    if P.shape[1] == 0:
        return P
    offset = np.array([[cx], [cy]])
    diff = np.abs(P[:2, :] - offset)
    valid = np.all(diff <= L, axis=0) & np.all(np.isfinite(P[:2, :]), axis=0)
    return P[:, valid]


# ---------------------------------------------------------------------------
# Polygon buffering and discretisation
# ---------------------------------------------------------------------------

def buffer_polygon(P, b, miter_limit=2.0):
    """Buffer a polygon by distance b.

    Uses Shapely if available, otherwise raises ImportError.

    Args:
        P:           (2, N) polygon boundary (may or may not be closed)
        b:           buffer distance (metres)
        miter_limit: miter limit for joint style (matches MATLAB polybuffer default)

    Returns:
        P_buf: (2, M) buffered polygon boundary (closed)
    """
    if not _SHAPELY:
        raise ImportError(
            "shapely is required for buffer_polygon. "
            "Install with: pip install shapely"
        )
    pts = list(zip(P[0], P[1]))
    poly = ShapelyPolygon(pts)

    # Try Shapely 2.x API first, fall back to 1.x
    try:
        buffered = poly.buffer(b, join_style='mitre', mitre_limit=miter_limit)
    except TypeError:
        buffered = poly.buffer(b, join_style=2)  # Shapely 1.x: 2 = mitre

    coords = np.array(buffered.exterior.coords).T  # (2, M)
    return coords


def interpolate_polyline_with_spacing(P, d):
    """Re-sample a polyline so that no segment is longer than d.

    Direct translation of RTD/utility/geometry/interpolate_polyline_with_spacing.m

    Args:
        P: (2, N) polyline (may contain NaN-column separators)
        d: maximum allowed point spacing

    Returns:
        O: (2, M) densely sampled polyline
    """
    if P.shape[1] < 2:
        return P.copy()

    # Check whether the polyline is closed
    closed = (P[0, 0] == P[0, -1] and P[1, 0] == P[1, -1]
              and not (np.isnan(P[0, 0]) or np.isnan(P[1, 0])))

    segments = []
    n = P.shape[1]

    for i in range(n - 1):
        p1 = P[:, i]
        p2 = P[:, i + 1]

        # Skip NaN segments
        if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
            # Keep the non-NaN endpoint if it exists
            if not np.any(np.isnan(p1)):
                segments.append(p1.reshape(2, 1))
            continue

        dist = np.linalg.norm(p2 - p1)
        if dist > d:
            # +1 matches MATLAB's ceil(dist/d)+1 to guarantee spacing <= d
            n_pts = int(np.ceil(dist / d)) + 1
            xs = np.linspace(p1[0], p2[0], n_pts)
            ys = np.linspace(p1[1], p2[1], n_pts)
            seg = np.vstack([xs, ys])
            segments.append(seg[:, :-1])  # drop last point to avoid duplication
        else:
            segments.append(p1.reshape(2, 1))

    if not segments:
        return np.zeros((2, 0))

    O = np.hstack(segments)

    # Close if needed
    if closed and not (O[0, 0] == O[0, -1] and O[1, 0] == O[1, -1]):
        O = np.hstack([O, O[:, :1]])

    return O


def make_random_polygon(n_vertices, center, scale):
    """Generate a random 2-D polygon (CCW, closed).

    Translation of simulator/src/utility/geometry/shape_creation/make_random_polygon.m

    Args:
        n_vertices: number of polygon vertices
        center:     (2,) or (2,1) array – polygon centre
        scale:      scalar – size of the polygon

    Returns:
        P: (2, n_vertices+1) polygon boundary (closed, CCW)
    """
    center = np.asarray(center, dtype=float).ravel()

    # Random vertices in [-0.5, 0.5]^2
    vertices = np.random.rand(2, n_vertices) - 0.5

    # Sort CCW around centroid (mirrors points_to_CCW.m)
    cx = vertices[0].mean()
    cy = vertices[1].mean()
    angles = np.arctan2(vertices[1] - cy, vertices[0] - cx)
    idx = np.argsort(angles)
    vertices = vertices[:, idx]

    # Close loop, scale, offset
    vertices = np.hstack([vertices, vertices[:, :1]])
    P = scale * vertices + center.reshape(2, 1)

    return P


# ---------------------------------------------------------------------------
# Point spacing (turtlebot footprint)
# ---------------------------------------------------------------------------

def compute_turtlebot_point_spacing(R, b):
    """Compute the obstacle-point spacing for a circular robot footprint.

    Translation of
    RTD_tutorial/step_4_online_planning/functions/compute_turtlebot_point_spacings.m

    Args:
        R: robot footprint radius (m)
        b: obstacle buffer distance (m)

    Returns:
        r: maximum allowed point spacing (m)
    """
    bbar = R
    if b > bbar:
        print('Resizing obstacle buffer to be valid!')
        b = bbar - 0.01

    theta_1 = np.arccos((R - b) / R)
    r = 2 * R * np.sin(theta_1)
    return r


# ---------------------------------------------------------------------------
# Full discretised-obstacle pipeline
# ---------------------------------------------------------------------------

def compute_turtlebot_discretized_obs(O_world, turtlebot_pose, b, r, frs):
    """Buffer, discretize, and transform obstacle to the FRS frame.

    Translation of
    RTD_tutorial/step_4_online_planning/functions/compute_turtlebot_discretized_obs.m

    Args:
        O_world:        (2, N) obstacle polygon in world frame
        turtlebot_pose: (3+,) robot state [x, y, heading, ...]
        b:              buffer distance (m)
        r:              point spacing (m)
        frs:            dict with keys 'initial_x', 'initial_y', 'distance_scale'

    Returns:
        O_FRS:  (2, M) obstacle points in FRS frame (inside unit box)
        O_buf:  (2, K) buffered obstacle boundary
        O_pts:  (2, L) discretized obstacle points in world frame
    """
    O_buf = buffer_polygon(O_world, b, miter_limit=2.0)
    O_pts = interpolate_polyline_with_spacing(O_buf, r)

    x0 = float(frs['initial_x'])
    y0 = float(frs['initial_y'])
    D  = float(frs['distance_scale'])

    O_FRS = world_to_FRS(O_pts, turtlebot_pose, x0, y0, D)
    O_FRS = crop_points_outside_region(0.0, 0.0, O_FRS, 1.0)

    return O_FRS, O_buf, O_pts
