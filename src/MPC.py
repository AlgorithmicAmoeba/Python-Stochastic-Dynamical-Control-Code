
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


def mpc_var(adjmean, fcovar, N, A, B, b, aline, bline, cline, QQ, RR,
            ysp, usp, limu, limstepu, revconstr, swapcon, Q, sigma, growvar, d=numpy.zeros(2)):
    """return the MPC control input using a linear system"""

    # m = Model(solver=IpoptSolver(print_level=0)) # chooses optimiser by itself

    x = numpy.zeros([2, N])
    u = numpy.zeros(N - 1)
    B = B.T
    nx, nu = B.shape
    numpy.set_printoptions(linewidth=150)
    QN = QQ
    d_T = numpy.matrix(numpy.hstack([aline, bline]))

    P = scipy.sparse.block_diag([scipy.sparse.kron(scipy.sparse.eye(N), QQ), QN,
                                 scipy.sparse.kron(scipy.sparse.eye(N), RR)])

    q = numpy.hstack([numpy.kron(numpy.ones(N), -QQ @ ysp), -QN @ ysp,
                      numpy.kron(numpy.ones(N), -RR @ usp)])

    #Handling of mu_(k+1) = A @ mu_k + B @ u_k
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

    # Handling of -limu <= u <= limu
    temp1 = numpy.zeros([N, (N+1)*nx])
    temp2 = scipy.sparse.kron(scipy.sparse.eye(N), numpy.eye(nu))
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

    """# - linear dynamics
    Ax = scipy.sparse.kron(scipy.sparse.eye(N + 1), -scipy.sparse.eye(nx))
    Ax += scipy.sparse.kron(scipy.sparse.eye(N + 1, k=-1), A)
    Bu = scipy.sparse.kron(scipy.sparse.vstack([scipy.sparse.csc_matrix((1, N)), scipy.sparse.eye(N)]), B)
    Aeq = scipy.sparse.hstack([Ax, Bu])
    leq = numpy.hstack([-adjmean, numpy.zeros(N * nx)])

    # # add distribution constraints (Ipopt doens't like it when its a quadratic constraint because nonconvex)
    # sigma = 2.2788 # one sigma 68 % confidence
    # sigma = 4.605 # 90 % confidence
    # sigma = 9.21 # 99 % confidence
    constraints = scipy.sparse.hstack([scipy.sparse.kron(scipy.sparse.eye(N), d), numpy.zeros([N, nu])])
    u_constaraints = numpy.hstack([numpy.zeros([N, nx*N]), numpy.eye(N)])
    C_T = scipy.sparse.vstack([Aeq, constraints, u_constaraints])"""
    e = -cline
    if growvar:
        sigmas = Q + A @ fcovar @ A.T
        limits = numpy.zeros(N)
        for k in range(N):  # don't do anything about k=1
            rsquared = d_T @ sigmas @ d_T.T
            r = (numpy.sqrt(sigma*rsquared)-e)*swapcon
            sigmas = Q + A @ sigmas @ A.T
            limits[k] = r
    else:
        sigmas = Q + A @ fcovar @ A.T
        limits = numpy.zeros(N)
        for k in range(N):  # don't do anything about k=1
            rsquared = d_T @ sigmas @ d_T.T
            r = (numpy.sqrt(sigma*rsquared) - e) * swapcon
            limits[k] = r
    L = scipy.sparse.hstack([-adjmean, numpy.zeros(N*nx), limits, [-limu]*N, [-limstepu]*(N-1)])
    U = scipy.sparse.hstack([-adjmean, numpy.zeros(N*nx), [numpy.inf]*N, [limu]*N, [limstepu]*(N-1)])
    """
    lower = numpy.hstack([leq, limits, [-limu]*N])
    upper = numpy.hstack([leq, [numpy.inf]*len(limits), [limu]*N])"""
    prob = osqp.OSQP()

    nullwrite = NullWriter()
    oldstdout = sys.stdout
    sys.stdout = nullwrite  # disable output
    #prob.update_settings(verbose=False)
    prob.setup(P, q, AA, L.todense().T, U.todense().T, warm_start=True)


    res = prob.solve()

    sys.stdout = oldstdout  # enable output

    if res.info.status != 'solved':
        raise ValueError('OSQP did not solve the problem!')
    print(res.x[(N+1)*nx-2: (N+1)*nx+nu+2])
    return res.x[(N+1)*nx: (N+1)*nx+nu]
    #status = solve(m)

    # return getValue(u[1]) # get the controller input


"""
function mpc_lqr(adjmean, horizon, A, B, b, QQ, RR, ysp, usp, d=zeros(2))
  # return the MPC control input using a linear system

  # m = Model(solver=IpoptSolver(print_level=0)) # chooses optimiser by itself
  m = Model(solver=MosekSolver(LOG=0)) # chooses optimiser by itself

  @defVar(m, x[1:2, 1:horizon])
  @defVar(m, u[1:horizon-1])

  @addConstraint(m, x[1, 1] == adjmean[1])
  @addConstraint(m, x[2, 1] == adjmean[2])
  @addConstraint(m, x[1, 2] == A[1,1]*adjmean[1] + A[1,2]*adjmean[2] + B[1]*u[1] + d[1])
  @addConstraint(m, x[2, 2] == A[2,1]*adjmean[1] + A[2,2]*adjmean[2] + B[2]*u[1] + d[2])

  for k=3:horizon
    @addConstraint(m, x[1, k] == A[1,1]*x[1, k-1] + A[1,2]*x[2, k-1] + B[1]*u[k-1] + d[1])
    @addConstraint(m, x[2, k] == A[2,1]*x[1, k-1] + A[2,2]*x[2, k-1] + B[2]*u[k-1] + d[2])
  end


  @setObjective(m, Min, sum{QQ[1]*x[1, i]^2 - 2.0*ysp*QQ[1]*x[1, i] + RR*u[i]^2 - 2.0*usp*RR*u[i], i=1:horizon-1} + QQ[1]*x[1, horizon]^2 - 2.0*QQ[1]*ysp*x[1, horizon])

  status = solve(m)

  if status != :Optimal
    warn("Mosek did not converge. Attempting to use Ipopt...")
    unow = mpc_lqr_i(adjmean, horizon, A, B, b, QQ, RR, ysp, usp, d)
    return unow
  end

  return getValue(u[1]) # get the controller input
end
"""