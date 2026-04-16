import numpy as np
from .params import default_params

# Toggle Coriolis/centrifugal term.
# If you see instability or performance issues, set this to False.
USE_CORIOLIS = False

def _cum_angles(q):
    return np.cumsum(q)

def com_positions_xz(q, p):
    """
    COM positions in XZ plane for each link.
    Returns array shape (4,2): [ [x_c1,z_c1], ...]
    Convention:
      x = sum l*cos(th), z = sum l*sin(th)
      th_i = q1+...+qi
    """
    l = p["l"]
    lc = p["lc"]

    th = _cum_angles(q)

    com = np.zeros((4, 2), dtype=float)

    # joint i position (start of link i)
    xj = 0.0
    zj = 0.0

    for i in range(4):
        com[i, 0] = xj + lc[i] * np.cos(th[i])
        com[i, 1] = zj + lc[i] * np.sin(th[i])

        # move to next joint (end of link i)
        xj = xj + l[i] * np.cos(th[i])
        zj = zj + l[i] * np.sin(th[i])

    return com

def Jv_com_xz(i, q, p):
    """
    Linear velocity Jacobian (XZ) for COM of link i.
    Returns 2x4 matrix J so that [dx; dz] = J @ dq.
    """
    l = p["l"]
    lc = p["lc"]
    th = _cum_angles(q)

    J = np.zeros((2, 4), dtype=float)

    # p_com_i = sum_{r=0..i-1} l[r]*[cos(th[r]), sin(th[r])] + lc[i]*[cos(th[i]), sin(th[i])]
    for k in range(4):
        if k > i:
            continue

        dx = 0.0
        dz = 0.0

        # full link contributions up to i-1
        for r in range(k, i):
            dx += -l[r] * np.sin(th[r])
            dz +=  l[r] * np.cos(th[r])

        # COM segment on link i
        dx += -lc[i] * np.sin(th[i])
        dz +=  lc[i] * np.cos(th[i])

        J[0, k] = dx
        J[1, k] = dz

    return J

def Jw_link(i):
    """
    Angular velocity Jacobian for link i (planar about joint axis).
    omega_i = dq1 + ... + dq_{i}
    Returns shape (4,)
    """
    j = np.zeros(4, dtype=float)
    j[: i + 1] = 1.0
    return j

def mass_matrix(q, p):
    """
    M(q) = sum_i m_i Jv_i^T Jv_i + I_i Jw_i^T Jw_i
    """
    m = p["m"]
    I = p["I"]  # planar inertia about joint axis through COM (your rod approx)

    M = np.zeros((4, 4), dtype=float)
    for i in range(4):
        Jv = Jv_com_xz(i, q, p)           # 2x4
        Jw = Jw_link(i).reshape(1, 4)     # 1x4
        M += m[i] * (Jv.T @ Jv) + I[i] * (Jw.T @ Jw)

    # numerical symmetry cleanup
    return 0.5 * (M + M.T)

def potential_energy(q, p):
    """
    V = sum_i m_i * g0 * z_com_i
    (+z is up; gravity points -z)
    """
    m = p["m"]
    g0 = p["g0"]
    com = com_positions_xz(q, p)
    z = com[:, 1]
    return float(np.sum(m * g0 * z))

def gravity_vector(q, p, eps=1e-7):
    """
    g(q) = dV/dq using central differences.
    """
    g = np.zeros(4, dtype=float)
    for k in range(4):
        dqk = np.zeros(4, dtype=float)
        dqk[k] = eps
        Vp = potential_energy(q + dqk, p)
        Vm = potential_energy(q - dqk, p)
        g[k] = (Vp - Vm) / (2.0 * eps)
    return g

def coriolis_vector(q, dq, p, eps=1e-6):
    """
    c(q,dq) = C(q,dq) dq computed from Christoffel symbols using numeric dM/dq.
    """
    dq = np.asarray(dq, dtype=float)

    # dM[k] = dM/dq_k
    dM = np.zeros((4, 4, 4), dtype=float)
    for k in range(4):
        dqk = np.zeros(4, dtype=float)
        dqk[k] = eps
        Mp = mass_matrix(q + dqk, p)
        Mm = mass_matrix(q - dqk, p)
        dM[k] = (Mp - Mm) / (2.0 * eps)

    c = np.zeros(4, dtype=float)
    for j in range(4):
        s = 0.0
        for k in range(4):
            for l in range(4):
                Gamma = 0.5 * (dM[l][j, k] + dM[k][j, l] - dM[j][k, l])
                s += Gamma * dq[k] * dq[l]
        c[j] = s
    return c

def ddq_rigid(q, dq, tau):
    """
    Full rigid-body vertical-plane dynamics with damping:
      M(q) ddq + c(q,dq) + g(q) + D dq = tau
    """
    p = default_params()
    D = p["D"]

    q = np.asarray(q, dtype=float)
    dq = np.asarray(dq, dtype=float)
    tau = np.asarray(tau, dtype=float)

    M = mass_matrix(q, p)
    g = gravity_vector(q, p)

    if USE_CORIOLIS:
        c = coriolis_vector(q, dq, p)
    else:
        c = np.zeros(4, dtype=float)

    rhs = tau - c - g - (D @ dq)

    # Solve instead of inverse
    ddq = np.linalg.solve(M, rhs)
    return ddq