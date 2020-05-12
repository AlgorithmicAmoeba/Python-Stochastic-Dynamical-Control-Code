# Controller using the linear reactor model measuring both concentration and temperature.

import closedloop_scenarios_single.closedloop_params
import src.LQR as LQR
import src.LLDS as LLDS
import scipy.stats
import src.MPC as MPC
import src.Results as Results
import numpy
import matplotlib.pyplot as plt

tend = 80
params = closedloop_scenarios_single.closedloop_params.Params(tend)  # end time of simulation

# Get the linear model
linsystems = params.cstr_model.get_nominal_linear_systems(params.h)  # cstr_model comes from params.jl
opoint = 1  # the specific operating point we are going to use for control

init_state = numpy.array([0.55, 450])  # random initial point near operating point

# Set the state space model
A = linsystems[opoint].A
B = linsystems[opoint].B
b = linsystems[opoint].b  # offset from the origin

# Set point
ysp = linsystems[opoint].op[0] - b[0]  # Medium concentration
H = numpy.matrix([1, 0])  # only attempt to control the concentration
x_off, usp = LQR.offset(A, numpy.matrix(B), params.C2, H, numpy.matrix([ysp]))  # control offset
ysp = x_off + numpy.array([-0.1, 0])
usp = numpy.array([usp])

# Set up the KF
kf_cstr = LLDS.LLDS(A, B, params.C2, params.Q, params.R2)  # set up the KF object (measuring both states)
state_noise_dist = scipy.stats.multivariate_normal(cov=params.Q)
meas_noise_dist = scipy.stats.multivariate_normal(cov=params.R2)

# First time step of the simulation
params.xs[:, 0] = init_state - b  # set simulation starting point to the random initial state
params.ys2[:, 0] = params.C2 @ params.xs[:, 0] + meas_noise_dist.rvs()  # measure from actual plant
temp = kf_cstr.init_filter(init_state-b, params.init_state_covar, params.ys2[:, 0]-b)  # filter
params.kfmeans[:, 0], params.kfcovars[:, :, 0] = temp

horizon = 150
params.us[0] = MPC.mpc_lqr(params.xs[:, 0], horizon, A, numpy.matrix(B),
                           params.QQ, params.RR, ysp, numpy.array([0.0]))  # get the controller input

for t in range(1, params.N):
    params.xs[:, t] = params.cstr_model.run_reactor(params.xs[:, t-1] + b, params.us[t-1], params.h) - b
    params.xs[:, t] += state_noise_dist.rvs()

    params.ys2[:, t] = params.C2 @ params.xs[:, t] + meas_noise_dist.rvs()  # measure from actual plant
    temp = kf_cstr.step_filter(params.kfmeans[:, t - 1], params.kfcovars[:, :, t - 1], params.us[t - 1],
                               params.ys2[:, t])
    params.kfmeans[:, t], params.kfcovars[:, :, t] = temp

    # Compute controller action
    if t % 10 == 0:
        params.us[t] = MPC.mpc_lqr(params.xs[:, t], horizon, A,
                                   numpy.matrix(B), params.QQ, params.RR,
                                   ysp, numpy.array([0.0]))  # get the controller input
        if params.us[t] is None or numpy.isnan(params.us[t]):
            break
    else:
        params.us[t] = params.us[t - 1]

for i in range(len(params.kfmeans[0])):
    params.kfmeans[:, i] += b
    params.xs[:, i] += b
    params.ys2[:, i] += b

# Plot the results
Results.plot_tracking1(params.ts, params.xs, params.ys2, params.kfmeans, params.us, 2, ysp[0] + b[0])
Results.calc_error1(params.xs, ysp[0] + b[0])
Results.calc_energy(params.us, 0.0)
plt.show()
