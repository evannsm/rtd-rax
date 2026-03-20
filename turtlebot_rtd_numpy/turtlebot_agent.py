"""
turtlebot_agent.py  –  NumPy version
======================================
Turtlebot unicycle-dynamics agent with state tracking.

Translation of RTD_tutorial/simulator_files/agent/turtlebot_agent.m
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class TurtlebotAgent:
    """Unicycle-model Turtlebot with 4-state dynamics [x, y, heading, v].

    Inputs are [w (yaw rate), a (acceleration)].
    """

    # Physical limits
    footprint   = 0.35 / 2   # radius (m)
    max_speed   = 2.0         # m/s
    max_yaw_rate = 2.0        # rad/s
    max_accel   = 2.0         # m/s²

    def __init__(self):
        self.state = None        # (4, N_total) – full trajectory history
        self.time  = None        # (N_total,)   – time history

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def reset(self, z0):
        """Set initial state.

        Args:
            z0: array-like of length 4  [x, y, heading, v]
        """
        z0 = np.asarray(z0, dtype=float).ravel()
        assert len(z0) == 4, "State must be [x, y, heading, v]"
        self.state = z0.reshape(4, 1)
        self.time  = np.array([0.0])

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def dynamics(self, t, z, T_ref, U_ref, Z_ref, disturbance=None):
        """Unicycle ODE RHS (closed-loop with simple feedforward LLC).

        The low-level controller uses the desired trajectory directly as
        feedforward:  w = w_des,  a = a_des  (no feedback on x/y/h).

        Args:
            t:     current time
            z:     (4,) state [x, y, h, v]
            T_ref: (N,) reference times
            U_ref: (2, N) reference inputs [w_des; a_des]
            Z_ref: (4, N) reference states (unused here)
            disturbance: optional callable `(t, z) -> [dpx, dpy, dh, dv]`
                         returning additive disturbance on the state derivative

        Returns:
            zd: (4,) state derivative
        """
        h = z[2]
        v = z[3]

        # Interpolate desired inputs
        w_des = float(np.interp(t, T_ref, U_ref[0, :]))
        a_des = float(np.interp(t, T_ref, U_ref[1, :]))

        # Saturate inputs
        w = np.clip(w_des, -self.max_yaw_rate, self.max_yaw_rate)
        a = np.clip(a_des, -self.max_accel,   self.max_accel)

        # Unicycle derivatives
        xd = v * np.cos(h)
        yd = v * np.sin(h)
        hd = w
        vd = a

        if disturbance is None:
            d = np.zeros(4, dtype=float)
        else:
            d = np.asarray(disturbance(t, z), dtype=float).ravel()
            if d.shape != (4,):
                raise ValueError('disturbance(t, z) must return a length-4 vector')

        return [xd + d[0], yd + d[1], hd + d[2], vd + d[3]]

    # ------------------------------------------------------------------
    # Move
    # ------------------------------------------------------------------

    def move(self, t_move, T_ref, U_ref, Z_ref, disturbance=None):
        """Integrate the unicycle dynamics forward in time.

        Args:
            t_move: total time to move (s)
            T_ref:  (N,) reference times
            U_ref:  (2, N) reference inputs
            Z_ref:  (4, N) reference states
            disturbance: optional callable `(t, z) -> [dpx, dpy, dh, dv]`
                         returning additive disturbance on the state derivative
        """
        z0 = self.state[:, -1].copy()
        t0 = float(self.time[-1])
        t_span = (t0, t0 + t_move)

        # Re-centre reference times to start at t0
        T_shifted = T_ref - T_ref[0] + t0

        sol = solve_ivp(
            fun=lambda t, z: self.dynamics(t, z, T_shifted, U_ref, Z_ref, disturbance),
            t_span=t_span,
            y0=z0,
            method='RK45',
            t_eval=np.linspace(t0, t0 + t_move, len(T_ref)),
            rtol=1e-6,
            atol=1e-8,
        )

        # Append to history (skip duplicate initial point)
        self.state = np.hstack([self.state, sol.y[:, 1:]])
        self.time  = np.append(self.time, sol.t[1:])

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, ax=None, color='b', alpha=0.3):
        """Draw the robot footprint circle and travelled path.

        Args:
            ax:    matplotlib Axes (uses current axes if None)
            color: colour for the robot fill
            alpha: transparency of footprint circle
        """
        if ax is None:
            ax = plt.gca()

        # Current position
        x, y = float(self.state[0, -1]), float(self.state[1, -1])

        # Footprint circle
        circle = mpatches.Circle((x, y), self.footprint,
                                  color=color, alpha=alpha, zorder=2)
        ax.add_patch(circle)

        # Heading arrow
        h = float(self.state[2, -1])
        ax.annotate('', xy=(x + self.footprint * np.cos(h),
                             y + self.footprint * np.sin(h)),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color='k', lw=1.5))

        # Path
        if self.state.shape[1] > 1:
            ax.plot(self.state[0, :], self.state[1, :],
                    color=color, linewidth=1.5, zorder=1, label='robot path')

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def pose(self):
        """Current pose [x, y, heading]."""
        return self.state[:3, -1]

    @property
    def speed(self):
        """Current speed (m/s)."""
        return float(self.state[3, -1])
