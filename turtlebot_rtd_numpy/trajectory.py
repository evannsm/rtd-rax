"""
trajectory.py  –  NumPy / SciPy version
=========================================
Trajectory-producing model and braking trajectory for the Turtlebot.

Direct translations of:
  RTD_tutorial/step_1_desired_trajectories/functions/turtlebot_trajectory_producing_model.m
  RTD_tutorial/step_1_desired_trajectories/functions/make_turtlebot_braking_trajectory.m
  RTD_tutorial/step_1_desired_trajectories/functions/get_turtlebot_braking_scale_at_time.m
"""

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Braking scale
# ---------------------------------------------------------------------------

def get_braking_scale(t, t_plan, t_stop, pwr=1):
    """Scalar braking scale ∈ [0,1] for t ≥ t_plan.

    Translation of get_turtlebot_braking_scale_at_time.m

    Args:
        t:      scalar or array of time values
        t_plan: planning time horizon (s)
        t_stop: time required to brake to zero (s)
        pwr:    braking exponent (1 = linear)

    Returns:
        s: braking scale (same shape as t)
    """
    t = np.asarray(t, dtype=float)
    s = ((t_stop - t + t_plan) / t_stop) ** pwr
    return np.clip(s, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Trajectory-producing model (Dubins car for ODE integration)
# ---------------------------------------------------------------------------

def _traj_model_rhs(t, z, T_in, U_in):
    """RHS of the turtlebot trajectory-producing model.

    State z = [x, y, heading].
    Input u = [w_des, v_des] interpolated from (T_in, U_in).

    Translation of turtlebot_trajectory_producing_model.m
    """
    h = z[2]

    # Interpolate input at current time
    w_des = float(np.interp(t, T_in, U_in[0, :]))
    v_des = float(np.interp(t, T_in, U_in[1, :]))

    return [v_des * np.cos(h),
            v_des * np.sin(h),
            w_des]


# ---------------------------------------------------------------------------
# Braking trajectory
# ---------------------------------------------------------------------------

def make_turtlebot_braking_trajectory(t_plan, t_stop, w_des, v_des):
    """Create a full-state braking trajectory for the Turtlebot.

    Translation of make_turtlebot_braking_trajectory.m

    The robot follows a Dubins arc from t=0 to t=t_plan, then brakes
    linearly to rest over [t_plan, t_plan+t_stop].

    Args:
        t_plan: planning horizon (s)
        t_stop: duration for robot to stop (s)
        w_des:  desired yaw rate (rad/s)
        v_des:  desired linear speed (m/s)

    Returns:
        T: (N,)   time vector
        U: (2, N) inputs [w; a] (yaw-rate, acceleration)
        Z: (4, N) states [x; y; heading; v]
    """
    t_sample = 0.01
    t_total = t_plan + t_stop

    # Time vector matching MATLAB's unique([0:dt:t_total, t_total])
    T = np.arange(0.0, t_total + t_sample / 2, t_sample)
    if T[-1] != t_total:
        T = np.append(T, t_total)
    T = np.unique(T)

    # Braking scale
    scale = np.ones_like(T)
    braking_mask = T >= t_plan
    scale[braking_mask] = get_braking_scale(T[braking_mask], t_plan, t_stop, pwr=1)

    # Desired inputs
    w_traj = w_des * scale
    v_traj = v_des * scale
    U_in = np.vstack([w_traj, v_traj])  # (2, N)

    # Integrate trajectory-producing model from z0 = [0, 0, 0]
    sol = solve_ivp(
        fun=lambda t, z: _traj_model_rhs(t, z, T, U_in),
        t_span=(T[0], T[-1]),
        y0=[0.0, 0.0, 0.0],
        method='RK45',
        t_eval=T,
        rtol=1e-6,
        atol=1e-8,
    )

    Z_xyz = sol.y  # (3, N)  [x; y; heading]

    # Append velocity to create full 4-state trajectory
    Z = np.vstack([Z_xyz, v_traj])  # (4, N)

    # Compute acceleration input
    a_traj = np.diff(v_traj) / t_sample
    a_traj = np.append(a_traj, a_traj[-1])  # repeat last value
    U = np.vstack([w_traj, a_traj])  # (2, N)

    return T, U, Z
