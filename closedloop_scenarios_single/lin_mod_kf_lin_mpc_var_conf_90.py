# Linear Plant controlled with a linear MPC using a KF to estimate the state.
# Stochastic constraints.

import closedloop_scenarios_single.closedloop_params
import src.LQR as LQR
import src.LLDS as LLDS
import scipy.stats
import src.MPC as MPC
import src.Results as Results
import numpy

tend = 80
params = closedloop_scenarios_single.closedloop_params.Params(tend)  # end time of simulation

# Get the linear model
linsystems = params.cstr_model.get_nominal_linear_systems(params.h)  # cstr_model comes from params.jl
opoint = 1  # the specific operating point we are going to use for control

init_state = numpy.array([0.55, 450])  # random initial point near operating point

# Set the state space model
A = linsystems[opoint].A
B = numpy.matrix(linsystems[opoint].B)
b = linsystems[opoint].b  # offset from the origin

# Set point
ysp = linsystems[opoint].op[0] - b[0]  # Medium concentration
H = numpy.matrix([1, 0])  # only attempt to control the concentration
x_off, usp = LQR.offset(A, B, params.C2, H, numpy.array([ysp]))  # control offset
ysp = x_off
usp = numpy.array([usp])

# Set up the KF
kf_cstr = LLDS.LLDS(A, B, params.C2, params.Q, params.R2)  # set up the KF object (measuring both states)
state_noise_dist = scipy.stats.multivariate_normal(cov=params.Q)
meas_noise_dist = scipy.stats.multivariate_normal(cov=params.R2)

# First time step of the simulation
params.xs[:, 0] = init_state - b  # set simulation starting point to the random initial state
params.ys2[:, 0] = params.C2 @ params.xs[:, 0] + meas_noise_dist.rvs()  # measure from actual plant
temp = kf_cstr.init_filter(init_state-b, params.init_state_covar, params.ys2[:, 0])  # filter
params.kfmeans[:, 0], params.kfcovars[:, :, 0] = temp
# Setup MPC
horizon = 2
# add state constraints
aline = 10  # slope of constraint line ax + by + c = 0
cline = -412  # negative of the y axis intercept
bline = 1

params.us[0] = MPC.mpc_var(params.kfmeans[:, 0], params.kfcovars[:, :, 0], horizon,
                           A, B, b, aline, bline, cline, params.QQ, params.RR, ysp,
                           usp[0], 10000.0, 1000.0, False, 1.0, params.Q, 4.6052, True)  # get the controller input
for t in range(1, params.N):
    params.xs[:, t] = A @ params.xs[:, t-1] + B*params.us[t-1] + state_noise_dist.rvs()  # actual plant
    params.ys2[:, t] = params.C2 @ params.xs[:, t] + meas_noise_dist.rvs()  # measure from actual plant
    temp = kf_cstr.step_filter(params.kfmeans[:, t-1], params.kfcovars[:, :, t-1], params.us[t-1], params.ys2[:, t])
    params.kfmeans[:, t], params.kfcovars[:, :, t] = temp

    if t % 10 == 0:
        params.us[t] = MPC.mpc_var(params.kfmeans[:, t], params.kfcovars[:, :, t], horizon,
                                   A, B, b, aline, bline, cline, params.QQ, params.RR, ysp,
                                   usp[0], 10000.0, 1000.0, False, 1.0, params.Q, 4.6052, True)  # get controller input
    else:
        params.us[t] = params.us[t-1]

for i in range(len(params.kfmeans)):
    params.kfmeans += b
    params.xs += b
    params.ys2 += b

# Plot the results
Results.plot_tracking1(params.ts, params.xs, params.ys2, params.kfmeans, params.us, 2, ysp+b[1])
Results.plot_ellipses2(params.ts, params.xs, params.kfmeans, params.kfcovars, [aline, cline],
                       linsystems[2].op, True, 4.6052, 1, "best")
Results.check_constraint(params.ts, params.xs, [aline, cline])
Results.calc_error(params.xs, ysp+b[1])
Results.calc_energy(params.us, 0.0)
