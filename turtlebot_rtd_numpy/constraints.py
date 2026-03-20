"""
constraints.py  –  NumPy version
==================================
Nonlinear constraint function and Jacobian for the Turtlebot RTD optimizer.

Translation of:
  RTD_tutorial/step_4_online_planning/functions/turtlebot_nonlcon_for_fmincon.m

The constraint is:   FRS_polynomial(z, k) - 1 ≤ 0   for each obstacle point z.
After substituting z-values this becomes an N-vector polynomial in k.

For scipy.optimize with method='SLSQP' (or 'trust-constr'), inequality
constraints use the convention  g(k) ≥ 0.  So we negate the polynomial:
    g(k) = -(cons_poly(k)) ≥ 0   ⟺   cons_poly(k) ≤ 0  ✓
"""

import numpy as np
from polynomial_utils import eval_constraint_poly, eval_constraint_gradient


def build_constraint(cons_poly_struct, cons_grad_struct):
    """Create scipy-compatible constraint dict for SLSQP / trust-constr.

    Args:
        cons_poly_struct: dict from evaluate_frs_polynomial_on_obstacle_points
        cons_grad_struct: dict from get_constraint_polynomial_gradient

    Returns:
        constraint: dict with keys 'type', 'fun', 'jac'
    """
    def fun(k):
        # c ≤ 0 is safe; scipy 'ineq' requires g ≥ 0, so return -c
        return -eval_constraint_poly(cons_poly_struct, k)

    def jac(k):
        # Jacobian of -c: (N, 2) array
        return -eval_constraint_gradient(cons_grad_struct, k)

    return {'type': 'ineq', 'fun': fun, 'jac': jac}


def turtlebot_nonlcon(k, cons_poly_struct, cons_grad_struct):
    """Evaluate constraint values and Jacobian (MATLAB fmincon style).

    Returns the same four outputs as turtlebot_nonlcon_for_fmincon.m:
        n    – (N,) inequality constraint values  (feasible iff all ≤ 0)
        neq  – [] (no equality constraints)
        gn   – (N, 2) Jacobian of n w.r.t. k  (columns are ∂n/∂k_i)
        gneq – [] (no equality constraints)

    Args:
        k:                 (2,) trajectory parameter vector
        cons_poly_struct:  dict from evaluate_frs_polynomial_on_obstacle_points
        cons_grad_struct:  dict from get_constraint_polynomial_gradient

    Returns:
        n, neq, gn, gneq
    """
    n    = eval_constraint_poly(cons_poly_struct, k)
    neq  = np.array([])
    gn   = eval_constraint_gradient(cons_grad_struct, k)   # (N, 2)
    gneq = np.array([])
    return n, neq, gn, gneq
