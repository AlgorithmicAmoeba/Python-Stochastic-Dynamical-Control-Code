
import osqp
import numpy
import scipy.sparse
import sys


class NullWriter(object):
    def write(self, arg):
        pass


print("MPC is hardcoded for the CSTR!")


def mpc_mean(x0, N, A, B, aline, bline, e, QQ, RR, ysp, usp, lim_u, lim_step_u):
    """return the MPC control input using a linear system"""
    B = B.T
    nx, nu = B.shape
    QN = QQ
    d_T = numpy.matrix(numpy.hstack([aline, bline]))

    P = scipy.sparse.block_diag([scipy.sparse.kron(scipy.sparse.eye(N), QQ), QN,
                                 scipy.sparse.kron(scipy.sparse.eye(N), RR)])

    q = numpy.hstack([numpy.kron(numpy.ones(N), -QQ @ ysp), -QN @ ysp,
                      numpy.kron(numpy.ones(N), -RR @ usp)])

    # Handling of mu_(k+1) = A @ mu_k + B @ u_k
    temp1 = scipy.sparse.block_diag([scipy.sparse.kron(scipy.sparse.eye(N + 1), -numpy.eye(nx))])
    temp2 = scipy.sparse.block_diag([scipy.sparse.kron(scipy.sparse.eye(N + 1, k=-1), A)])
    AA = temp1 + temp2

    temp1 = scipy.sparse.vstack([numpy.zeros([nx, N * nu]), scipy.sparse.kron(scipy.sparse.eye(N), B)])
    AA = scipy.sparse.hstack([AA, temp1])

    # Handling of d.T mu_k > k sqrt(d.T @ Sigma_k @ d) - e
    temp1 = scipy.sparse.hstack([numpy.zeros([N, nx]), scipy.sparse.kron(scipy.sparse.eye(N), d_T)])
    temp2 = numpy.zeros([N, N * nu])
    temp3 = scipy.sparse.hstack([temp1, temp2])
    AA = scipy.sparse.vstack([AA, temp3])

    # Handling of -limstep <= u <= limstepu
    temp1 = numpy.zeros([N - 1, (N + 1) * nx])
    temp2 = numpy.zeros([N - 1, nu])
    temp2[0][:nu] = 1
    temp3 = scipy.sparse.kron(scipy.sparse.eye(N - 1), -numpy.eye(nu))
    temp3 += scipy.sparse.kron(scipy.sparse.eye(N - 1, k=-1), numpy.eye(nu))
    temp4 = scipy.sparse.hstack([temp1, temp2, temp3])
    AA = scipy.sparse.vstack([AA, temp4])

    # Handling of -limu <= u <= limu
    temp1 = numpy.zeros([N, (N + 1) * nx])
    temp2 = scipy.sparse.kron(scipy.sparse.eye(N), numpy.eye(nu))
    temp3 = scipy.sparse.hstack([temp1, temp2])
    AA = scipy.sparse.vstack([AA, temp3])

    e = -e
    limits = [-e] * N

    L = scipy.sparse.hstack([-x0, numpy.zeros(N * nx), limits, [-lim_step_u] * (N - 1), [-lim_u] * N])
    U = scipy.sparse.hstack([-x0, numpy.zeros(N * nx), [numpy.inf] * N, [lim_step_u] * (N - 1), [lim_u] * N])

    prob = osqp.OSQP()

    nullwrite = NullWriter()
    oldstdout = sys.stdout
    sys.stdout = nullwrite  # disable output
    # prob.update_settings(verbose=False)  Does not work. Causes segfault
    prob.setup(P, q, AA, L.todense().T, U.todense().T, warm_start=True)

    res = prob.solve()

    sys.stdout = oldstdout  # enable output

    if res.info.status != 'solved':
        raise ValueError('OSQP did not solve the problem!')
    return res.x[(N + 1) * nx: (N + 1) * nx + nu]


def mpc_var(x0, cov0, N, A, B, aline, bline, e, QQ, RR,
            ysp, usp, lim_u, lim_step_u, Q, k, growvar=True):
    """return the MPC control input using a linear system"""

    B = B.T
    nx, nu = B.shape
    QN = QQ
    d_T = numpy.matrix(numpy.hstack([aline, bline]))

    P = scipy.sparse.block_diag([scipy.sparse.kron(scipy.sparse.eye(N), QQ), QN,
                                 scipy.sparse.kron(scipy.sparse.eye(N), RR)])

    q = numpy.hstack([numpy.kron(numpy.ones(N), -QQ @ ysp), -QN @ ysp,
                      numpy.kron(numpy.ones(N), -RR @ usp)])

    # Handling of mu_(k+1) = A @ mu_k + B @ u_k
    temp1 = scipy.sparse.block_diag([scipy.sparse.kron(scipy.sparse.eye(N+1), -numpy.eye(nx))])
    temp2 = scipy.sparse.block_diag([scipy.sparse.kron(scipy.sparse.eye(N+1, k=-1), A)])
    AA = temp1 + temp2

    temp1 = scipy.sparse.vstack([numpy.zeros([nx, N*nu]), scipy.sparse.kron(scipy.sparse.eye(N), B)])
    AA = scipy.sparse.hstack([AA, temp1])

    # Handling of d.T mu_k > k sqrt(d.T @ Sigma_k @ d) - e
    temp1 = scipy.sparse.hstack([numpy.zeros([N, nx]), scipy.sparse.kron(scipy.sparse.eye(N), d_T)])
    temp2 = numpy.zeros([N, N*nu])
    temp3 = scipy.sparse.hstack([temp1, temp2])
    AA = scipy.sparse.vstack([AA, temp3])

    # Handling of -limstep <= u <= limstepu
    temp1 = numpy.zeros([N-1, (N + 1) * nx])
    temp2 = numpy.zeros([N-1, nu])
    temp2[0][:nu] = 1
    temp3 = scipy.sparse.kron(scipy.sparse.eye(N-1), -numpy.eye(nu))
    temp3 += scipy.sparse.kron(scipy.sparse.eye(N-1, k=-1), numpy.eye(nu))
    temp4 = scipy.sparse.hstack([temp1, temp2, temp3])
    AA = scipy.sparse.vstack([AA, temp4])

    # Handling of -limu <= u <= limu
    temp1 = numpy.zeros([N, (N+1)*nx])
    temp2 = scipy.sparse.kron(scipy.sparse.eye(N), numpy.eye(nu))
    temp3 = scipy.sparse.hstack([temp1, temp2])
    AA = scipy.sparse.vstack([AA, temp3])

    e = -e

    sigmas = Q + A @ cov0 @ A.T
    limits = numpy.zeros(N)
    for i in range(N):  # don't do anything about i=1
        rsquared = d_T @ sigmas @ d_T.T
        r = numpy.sqrt(k * rsquared) - e
        if growvar:
            sigmas = Q + A @ sigmas @ A.T
        limits[i] = r

    L = scipy.sparse.hstack([-x0, numpy.zeros(N * nx), limits, [-lim_step_u] * (N - 1), [-lim_u] * N])
    U = scipy.sparse.hstack([-x0, numpy.zeros(N * nx), [numpy.inf] * N, [lim_step_u] * (N - 1), [lim_u] * N])

    prob = osqp.OSQP()

    nullwrite = NullWriter()
    oldstdout = sys.stdout
    sys.stdout = nullwrite  # disable output
    # prob.update_settings(verbose=False)  Does not work. Causes segfault
    prob.setup(P, q, AA, L.todense().T, U.todense().T, warm_start=True)

    res = prob.solve()

    sys.stdout = oldstdout  # enable output

    if res.info.status != 'solved':
        raise ValueError('OSQP did not solve the problem!')

    return res.x[(N+1)*nx: (N+1)*nx+nu]


def mpc_lqr(x0, N, A, B, QQ, RR, ysp, usp):
    """return the MPC control input using a linear system"""

    B = B.T
    nx, nu = B.shape
    QN = QQ

    P = scipy.sparse.block_diag([scipy.sparse.kron(scipy.sparse.eye(N), QQ), QN,
                                 scipy.sparse.kron(scipy.sparse.eye(N), RR)])

    q = numpy.hstack([numpy.kron(numpy.ones(N), -QQ @ ysp), -QN @ ysp,
                      numpy.kron(numpy.ones(N), -RR @ usp)])

    # Handling of mu_(k+1) = A @ mu_k + B @ u_k
    temp1 = scipy.sparse.block_diag([scipy.sparse.kron(scipy.sparse.eye(N + 1), -numpy.eye(nx))])
    temp2 = scipy.sparse.block_diag([scipy.sparse.kron(scipy.sparse.eye(N + 1, k=-1), A)])
    AA = temp1 + temp2

    temp1 = scipy.sparse.vstack([numpy.zeros([nx, N * nu]), scipy.sparse.kron(scipy.sparse.eye(N), B)])
    AA = scipy.sparse.hstack([AA, temp1])

    L = scipy.sparse.hstack([-x0, numpy.zeros(N * nx)])
    U = scipy.sparse.hstack([-x0, numpy.zeros(N * nx)])

    prob = osqp.OSQP()

    nullwrite = NullWriter()
    oldstdout = sys.stdout
    sys.stdout = nullwrite  # disable output
    # prob.update_settings(verbose=False)  Does not work. Causes segfault
    prob.setup(P, q, AA, L.todense().T, U.todense().T, warm_start=True)

    res = prob.solve()

    sys.stdout = oldstdout  # enable output

    if res.info.status != 'solved':
        raise ValueError('OSQP did not solve the problem!')
    return res.x[(N + 1) * nx: (N + 1) * nx + nu]
