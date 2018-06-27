# Control using multiple linear models and measuring both concentration and temperature

import numpy
import scipy.stats
import closedloop_scenarios_single.closedloop_params as params
import src.Results as Results
import src.RBPF as RBPF
import src.LQR as LQR
import src.MPC as MPC
import matplotlib.pyplot as plt
import typing

tend = 150
params = params.Params(tend)

# Get the three linear models about the nominal operating points
linsystems = params.cstr_model.get_nominal_linear_systems(params.h)

# Setup the RBPF
models, _ = RBPF.setup_rbpf(linsystems, params.C2, params.Q, params.R2)
A = numpy.array([[0.99, 0.01, 0.00],
                 [0.01, 0.98, 0.01],
                 [0.00, 0.01, 0.99]])
numModels = len(models)  # number of linear models (will be 3)
nP = 500  # number of particles

init_state = linsystems[1].op

sguess = RBPF.get_initial_switches(init_state, linsystems)  # prior switch distribution
particles = RBPF.init_rbpf(sguess, init_state, params.init_state_covar, 2, nP)

switchtrack = numpy.zeros([len(linsystems), params.N])
maxtrack = numpy.zeros([len(linsystems), params.N])

# Setup the controllers
setpoint = linsystems[2].op[0]
H = numpy.matrix([1.0, 0.0])
controllers = [None]*len(models)  # type: typing.List[LQR.Controller]
for k in range(len(models)):
    ysp = numpy.matrix(setpoint - models[k].b[0])  # set point is set here
    x_off, u_off = LQR.offset(models[k].A, models[k].B, params.C2, H, ysp)
    K = LQR.lqr(models[k].A, numpy.matrix(models[k].B), params.QQ, params.RR)
    controllers[k] = LQR.Controller(K, x_off, u_off)


state_noise_dist = scipy.stats.multivariate_normal(cov=params.Q)
meas_noise_dist = scipy.stats.multivariate_normal(cov=params.R2)
params.xs[:, 0] = init_state
params.ys2[:, 0] = params.C2 @ params.xs[:, 0] + meas_noise_dist.rvs()  # measured from actual plant

particles = RBPF.init_filter(particles, 0.0, params.ys2[:, 0], models)
params.rbpfmeans[:, 0], params.rbpfcovars[:, :, 0] = RBPF.get_ave_stats(particles)

for k in range(len(linsystems)):
    loc = numpy.where(particles.ss == k)[0]
    s = 0
    for l in loc:
        s += particles.ws[l]

    switchtrack[k, 0] = s

maxtrack[:, 0] = RBPF.get_max_track(particles, numModels)

# Controller Input
ind = numpy.argmax(maxtrack[:, 0])  # use this model and controller
horizon = 150

params.us[0] = MPC.mpc_lqr(params.rbpfmeans[:, 0] - models[ind].b, horizon, models[ind].A, numpy.matrix(models[ind].B),
                           numpy.matrix(params.QQ), numpy.matrix(params.RR),
                           controllers[ind].x_off, numpy.array([controllers[ind].u_off[0]]))
# Loop through the rest of time

for t in range(1, params.N):
    params.xs[:, t] = params.cstr_model.run_reactor(params.xs[:, t-1], params.us[t-1], params.h)
    params.xs[:, t] += state_noise_dist.rvs()  # actual plant
    params.ys2[:, t] = params.C2 @ params.xs[:, t] + meas_noise_dist.rvs()  # measured from actual plant
    RBPF.rbpf_filter(particles, params.us[t-1], params.ys2[:, t], models, A)
    params.rbpfmeans[:, t], params.rbpfcovars[:, :, t] = RBPF.get_ave_stats(particles)

    for k in range(len(linsystems)):
        loc = numpy.where(particles.ss == k)[0]
        s = 0
        for l in loc:
            s += particles.ws[l]
        switchtrack[k, t] = s

    maxtrack[:, t] = RBPF.get_max_track(particles, numModels)

    # Controller Input
    if t % 10 == 0:
        ind = numpy.argmax(maxtrack[:, t])  # use this model and controller
        params.us[t] = MPC.mpc_lqr(params.rbpfmeans[:, t] - models[ind].b, horizon, models[ind].A,
                                   numpy.matrix(models[ind].B),
                                   numpy.matrix(params.QQ), numpy.matrix(params.RR),
                                   controllers[ind].x_off, numpy.array([controllers[ind].u_off[0]]))
    else:
        params.us[t] = params.us[t-1]

# Plot results
Results.plot_state_space_switch(linsystems, params.xs)
Results.plot_switch_selection(numModels, maxtrack, params.ts, False)
plt.savefig("/home/ex/Documents/CSC/report/results/Figure_10-4_python.pdf", bbox_inches="tight")
Results.plot_switch_selection(numModels, switchtrack, params.ts, True)
Results.plot_tracking1(params.ts, params.xs, params.ys2, params.rbpfmeans, params.us, 2, setpoint)
plt.savefig("/home/ex/Documents/CSC/report/results/Figure_10-5_python.pdf", bbox_inches="tight")
plt.show()
