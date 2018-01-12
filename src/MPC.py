
import osqp
import numpy
import scipy.sparse
import sys


class NullWriter(object):
    def write(self, arg):
        pass


print("MPC is hardcoded for the CSTR!")
"""
def mpc_mean(adjmean, horizon, A, B, b, aline, bline, cline, QQ, RR, ysp, usp, limu, limstepu, revconstr, d=None):
    # return the MPC control input using a linear system
    if d == None:
        d = numpy.zeros(2)
    # m = Model(solver=MosekSolver(LOG=0)) # chooses optimiser by itself

    @defVar(m, x[1:2, 1:horizon])

    if limu == 0.0:
        @defVar(m, u[1:horizon-1])
    else
        @defVar(m, -limu <= u[1:horizon-1] <= limu)

    @addConstraint(m, x[1, 1] == adjmean[1])
    @addConstraint(m, x[2, 1] == adjmean[2])
    @addConstraint(m, x[1, 2] == A[1,1]*adjmean[1] + A[1,2]*adjmean[2] + B[1]*u[1] + d[1])
    @addConstraint(m, x[2, 2] == A[2,1]*adjmean[1] + A[2,2]*adjmean[2] + B[2]*u[1] + d[2])

    for k=3:horizon
        @addConstraint(m, x[1, k] == A[1,1]*x[1, k-1] + A[1,2]*x[2, k-1] + B[1]*u[k-1] + d[1])
        @addConstraint(m, x[2, k] == A[2,1]*x[1, k-1] + A[2,2]*x[2, k-1] + B[2]*u[k-1] + d[2])

    # # add state constraints
    if revconstr
        for k=2:horizon # can't do anything about k=1
            @addConstraint(m, aline*(x[1, k] + b[1]) + bline*(x[2, k] + b[2]) <= -1.0*cline)
    else
        for k=2:horizon # can't do anything about k=1
            @addConstraint(m, aline*(x[1, k] + b[1]) + bline*(x[2, k] + b[2]) >= -1.0*cline)

    for k=2:horizon-1
        @addConstraint(m, u[k]-u[k-1] <= limstepu)
        @addConstraint(m, u[k]-u[k-1] >= -limstepu)

    @setObjective(m, Min, sum{QQ[1]*x[1, i]^2 - 2.0*ysp*QQ[1]*x[1, i] + RR*u[i]^2 - 2.0*usp*RR*u[i], i=1:horizon-1} + QQ[1]*x[1, horizon]^2 - 2.0*QQ[1]*ysp*x[1, horizon])

    status = solve(m)

    if status != :Optimal
        warn("Mosek did not converge. Attempting to use Ipopt...")
        unow = mpc_mean_i(adjmean, horizon, A, B, b, aline, bline, cline, QQ, RR, ysp, usp, limu, limstepu, revconstr, d)
        return unow

    return getValue(u[1]) # get the controller input

"""


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
