"""
polynomial_utils.py  –  NumPy version
=======================================
Sparse-polynomial data structures and evaluation routines for FRS-based
trajectory optimization.

Direct translations of:
  RTD/utility/polynomial_evaluation/get_FRS_polynomial_structure.m
  RTD/utility/polynomial_evaluation/evaluate_FRS_polynomial_on_obstacle_points.m
  RTD/utility/polynomial_evaluation/get_constraint_polynomial_gradient.m
  RTD/utility/polynomial_evaluation/evaluate_constraint_polynomial.m
  RTD/utility/polynomial_evaluation/evaluate_constraint_polynomial_gradient.m

All polynomials are represented as plain NumPy arrays (no msspoly needed).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Build FRS polynomial structure from preproc data
# ---------------------------------------------------------------------------

def get_frs_polynomial_structure(pows, coef, z_cols, k_cols):
    """Return a dict describing the FRS polynomial in (z, k).

    Args:
        pows:   (Nterms, Nvars) monomial exponent matrix
        coef:   (Nterms,) coefficient vector  [corresponds to FRS_poly - 1]
        z_cols: (Nz,) 0-indexed column indices in pows for the z variables
        k_cols: (Nk,) 0-indexed column indices in pows for the k variables

    Returns:
        p_struct: dict with keys 'pows', 'coef', 'z_cols', 'k_cols'
    """
    return {
        'pows':   np.array(pows,   dtype=float),
        'coef':   np.array(coef,   dtype=float).ravel(),
        'z_cols': np.array(z_cols, dtype=int).ravel(),
        'k_cols': np.array(k_cols, dtype=int).ravel(),
    }


# ---------------------------------------------------------------------------
# Substitute obstacle z-points into the FRS polynomial → constraint polys in k
# ---------------------------------------------------------------------------

def evaluate_frs_polynomial_on_obstacle_points(p_struct, O_FRS):
    """Substitute each obstacle point z into the FRS polynomial.

    For each of the N obstacle points in O_FRS, this produces one polynomial
    in k by substituting the z-values into the FRS polynomial.

    Translation of evaluate_FRS_polynomial_on_obstacle_points.m

    Args:
        p_struct: dict from get_frs_polynomial_structure
        O_FRS:    (Nz, N) obstacle points in FRS coordinates  (Nz == 2)

    Returns:
        p_k_struct: dict with:
            'coef': (N, Nterms_k)  coefficients of the N constraint polynomials
            'pows': (Nterms_k, Nk) unique monomial powers in k
            'N':    int  number of constraint polynomials
            'k_cols': (Nk,)  (identity mapping; always 0..Nk-1 after collapse)
    """
    pows   = p_struct['pows']    # (Nterms, Nvars)
    coef   = p_struct['coef']    # (Nterms,)
    z_cols = p_struct['z_cols']  # (Nz,) 0-indexed
    k_cols = p_struct['k_cols']  # (Nk,) 0-indexed

    Nterms = pows.shape[0]
    N      = O_FRS.shape[1]   # number of obstacle points
    Nz     = len(z_cols)
    Nk     = len(k_cols)

    # Sub-matrix of powers for z-variables  (Nterms, Nz)
    sub_pows_z = pows[:, z_cols]   # shape (Nterms, Nz)

    # Evaluate z contribution for each term and each obstacle point
    # z_contrib[j, n] = prod_i( O_FRS[i, n] ^ sub_pows_z[j, i] )
    #   shape: (Nterms, N)
    #
    # Broadcast:  O_FRS[newaxis, :, :]  is (1, Nz, N)
    #             sub_pows_z[:, :, newaxis]  is (Nterms, Nz, 1)
    z_contrib = np.prod(
        O_FRS[np.newaxis, :, :] ** sub_pows_z[:, :, np.newaxis],
        axis=1
    )  # (Nterms, N)

    # Coefficient matrix before collapsing like terms
    # p_k_coef[n, j] = coef[j] * z_contrib[j, n]
    p_k_coef = coef[np.newaxis, :] * z_contrib.T   # (N, Nterms)

    # Sub-matrix of powers for k-variables  (Nterms, Nk)
    p_k_pows = pows[:, k_cols]   # (Nterms, Nk)

    # --- Collapse like terms (matching MATLAB's sortrows + unique logic) ---
    # Sort rows of p_k_pows lexicographically (primary key = col 0)
    i_s = np.lexsort(p_k_pows.T[::-1])          # stable lex sort by col 0 first
    p_k_pows_sorted = p_k_pows[i_s]
    p_k_coef_sorted = p_k_coef[:, i_s]          # (N, Nterms)

    # Find unique rows and their mapping indices
    p_k_pows_unique, inv = np.unique(
        p_k_pows_sorted, axis=0, return_inverse=True
    )                                             # (Nterms_unique, Nk)
    Nterms_unique = p_k_pows_unique.shape[0]

    # Sum coefficient columns that map to the same unique power row
    p_k_coef_collapsed = np.zeros((N, Nterms_unique))
    for i in range(Nterms_unique):
        mask = inv == i
        p_k_coef_collapsed[:, i] = p_k_coef_sorted[:, mask].sum(axis=1)

    return {
        'coef':   p_k_coef_collapsed,            # (N, Nterms_unique)
        'pows':   p_k_pows_unique,               # (Nterms_unique, Nk)
        'N':      N,
        'k_cols': np.arange(Nk),
    }


# ---------------------------------------------------------------------------
# Compute constraint polynomial Jacobian
# ---------------------------------------------------------------------------

def get_constraint_polynomial_gradient(p_k_struct):
    """Differentiate the N constraint polynomials with respect to k.

    Translation of get_constraint_polynomial_gradient.m

    Args:
        p_k_struct: dict from evaluate_frs_polynomial_on_obstacle_points

    Returns:
        J_struct: dict with:
            'coef': (N*Nk, Nterms_k)  Jacobian coefficients
            'pows': (Nterms_k, Nk*Nk) Jacobian monomial powers
            'N':    int
    """
    coef = p_k_struct['coef']  # (N, Nterms)
    pows = p_k_struct['pows']  # (Nterms, Nk)
    N    = p_k_struct['N']
    Nk   = pows.shape[1]

    # J_coef: for each k_idx, multiply each row of coef by pows[:, k_idx]
    #   J_coef[N*idx : N*(idx+1), :] = coef * pows[:, idx]   (broadcast)
    J_coef_parts = []
    for idx in range(Nk):
        # coef is (N, Nterms), pows[:, idx] is (Nterms,)
        J_coef_parts.append(coef * pows[np.newaxis, :, idx])  # (N, Nterms)
    J_coef = np.vstack(J_coef_parts)  # (N*Nk, Nterms)

    # Derivative powers: d/dk_idx  →  decrement power of k_idx by 1
    # but keep at 0 if already 0  (same as MATLAB: p + (p==0) - 1)
    dpdk_pows = pows.copy()  # (Nterms, Nk)
    for idx in range(Nk):
        p = pows[:, idx]
        dpdk_pows[:, idx] = p + (p == 0).astype(float) - 1.0

    # J_pows: for each k_idx, the powers are
    #   [pows[:, 0], ..., dpdk_pows[:, idx], ..., pows[:, Nk-1]]
    # stacked horizontally for all idx  →  (Nterms, Nk*Nk)
    J_pows_parts = []
    for idx in range(Nk):
        part = np.hstack([
            pows[:, :idx],           # (Nterms, idx)
            dpdk_pows[:, idx:idx+1], # (Nterms, 1)
            pows[:, idx+1:],         # (Nterms, Nk-idx-1)
        ])  # (Nterms, Nk)
        J_pows_parts.append(part)
    J_pows = np.hstack(J_pows_parts)  # (Nterms, Nk*Nk)

    return {
        'coef': J_coef,   # (N*Nk, Nterms)
        'pows': J_pows,   # (Nterms, Nk*Nk)
        'N':    N,
    }


# ---------------------------------------------------------------------------
# Evaluation at a specific k
# ---------------------------------------------------------------------------

def eval_constraint_poly(p_k_struct, k):
    """Evaluate the N constraint polynomials at k.

    Translation of evaluate_constraint_polynomial.m

    Args:
        p_k_struct: dict from evaluate_frs_polynomial_on_obstacle_points
        k: (Nk,) trajectory parameter vector

    Returns:
        out: (N,) constraint values  (> 0 means infeasible)
    """
    coef = p_k_struct['coef']  # (N, Nterms)
    pows = p_k_struct['pows']  # (Nterms, Nk)
    k    = np.asarray(k, dtype=float).ravel()

    # Evaluate each monomial:  mono[j] = prod_l( k[l]^pows[j, l] )
    mono = np.prod(k[np.newaxis, :] ** pows, axis=1)  # (Nterms,)
    return coef @ mono  # (N,)


def eval_constraint_gradient(J_struct, k):
    """Evaluate the N×Nk Jacobian of the constraint polynomials at k.

    Translation of evaluate_constraint_polynomial_gradient.m

    Args:
        J_struct: dict from get_constraint_polynomial_gradient
        k: (Nk,) trajectory parameter vector

    Returns:
        out: (N, Nk) Jacobian matrix
    """
    J_coef = J_struct['coef']  # (N*Nk, Nterms_J)
    J_pows = J_struct['pows']  # (Nterms_J, Nk*Nk)
    N      = J_struct['N']
    k      = np.asarray(k, dtype=float).ravel()
    Nk     = len(k)

    out = np.zeros((N, Nk))
    for idx in range(Nk):
        pows_idx = J_pows[:, idx * Nk:(idx + 1) * Nk]  # (Nterms_J, Nk)
        mono     = np.prod(k[np.newaxis, :] ** pows_idx, axis=1)  # (Nterms_J,)
        coef_idx = J_coef[N * idx:N * (idx + 1), :]               # (N, Nterms_J)
        out[:, idx] = coef_idx @ mono

    return out
